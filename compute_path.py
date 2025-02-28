from numba import njit, typed, types, prange
from numba.typed import List, Dict
from math import radians, sin, cos, sqrt, atan2
import numpy as np


@njit(fastmath=True)
def calculate_distances(lat, lon, lats, lons):
    R = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat)
    phi2 = np.radians(lats)
    dphi = phi2 - phi1
    dlambda = np.radians(lons - lon)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def find_closest_node(coords, nodes):
    lat, lon = map(float, coords.split(','))
    distances = calculate_distances(lat, lon, nodes[:, 1], nodes[:, 2])
    return np.int32(nodes[np.argmin(distances), 0])

@njit(fastmath=True, parallel=True)
def process_neighbors_optimized(dist_data: np.ndarray, speed_data: np.ndarray, slope_data: np.ndarray, neighbor_counts: np.ndarray, indices: np.ndarray, current_velocities: np.ndarray) -> tuple:
    """
    Optimized function to process neighbor data with distance, speed, slope, and velocity using Numba.
    """
    total_neighbors = np.sum(neighbor_counts)
    distances = np.empty(total_neighbors, dtype=np.float32)
    speeds = np.empty(total_neighbors, dtype=np.float32)
    slopes = np.empty(total_neighbors, dtype=np.float32)
    offsets = np.empty((len(neighbor_counts), 2), dtype=np.int32)

    current_offset = 0
    for i in range(len(neighbor_counts)):
        n_neighbors = neighbor_counts[i]
        if n_neighbors > 0:
            start_idx = indices[i]
            end_idx = start_idx + n_neighbors

            distances[current_offset:current_offset + n_neighbors] = dist_data[start_idx:end_idx]
            speeds[current_offset:current_offset + n_neighbors] = speed_data[start_idx:end_idx]
            slopes[current_offset:current_offset + n_neighbors] = slope_data[start_idx:end_idx]

            offsets[i, 0] = current_offset
            offsets[i, 1] = n_neighbors
            current_offset += n_neighbors
    return distances, speeds, slopes, total_neighbors, offsets


@njit(fastmath=True)
def distance_to_target(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0)**2
    return 2.0 * R * atan2(sqrt(a), sqrt(1.0 - a))


@njit(fastmath=True, parallel=True)
def batch_calculate_consumption(vehicle_data, distances, speeds, slopes, velocities):
    """
    Vectorized version of calculate_consumption for processing multiple road segments at once.
    Uses fixed-size chunks for Numba compatibility.
    """
    n = len(distances)
    segment_times = np.zeros(n, dtype=np.float32)
    segment_energies = np.zeros(n, dtype=np.float32)
    #energies_regenerated = np.zeros(n, dtype=np.float32)
    final_velocities = np.zeros(n, dtype=np.float32)

    weight = vehicle_data[1]
    aero_constant = vehicle_data[2]
    gravity_weight = vehicle_data[3]
    F_rolling = vehicle_data[4]
    efficiency_factor = vehicle_data[5]
    regen_efficiency = vehicle_data[6]
    auxillary_power = vehicle_data[7]
    max_acceleration = vehicle_data[8]
    max_deceleration = vehicle_data[9]

    # Process each segment in parallel
    for i in prange(n):
        # Initialize segment-specific variables
        speed_change_time = 0.0
        E_acc = 0.0
        E_regen = 0.0
        remaining_distance = distances[i]
        final_velocity = speeds[i]
        current_velocity = velocities[i]

        # Handle acceleration
        if speeds[i] > current_velocity:
            # Calculate achievable speed
            achievable_speed = np.sqrt(current_velocity**2 + 2 * max_acceleration * distances[i])

            if achievable_speed < final_velocity:
                final_velocity = achievable_speed

            # Compute acceleration time and energy
            speed_change_time = np.float32((final_velocity - current_velocity) / max_acceleration)
            E_acc = 0.5 * weight * (final_velocity**2 - current_velocity**2)

            # Update remaining distance
            distance_accelerated = np.float32((final_velocity**2 - current_velocity**2) / (2 * max_acceleration))
            remaining_distance = distances[i] - distance_accelerated

        # Handle deceleration
        elif speeds[i] < current_velocity:
            required_deceleration_distance = (current_velocity**2 - speeds[i]**2) / (2 * max_deceleration)

            if required_deceleration_distance > distances[i]:
                final_velocity = np.sqrt(current_velocity**2 - 2 * max_deceleration * distances[i])

            # Compute deceleration time and energy
            speed_change_time = np.float32((current_velocity - final_velocity) / max_deceleration)
            E_brake = 0.5 * weight * (current_velocity**2 - final_velocity**2)
            E_regen = regen_efficiency * E_brake

            # Update remaining distance
            distance_decelerated = np.float32((current_velocity**2 - final_velocity**2) / (2 * max_deceleration))
            remaining_distance = np.float32(distances[i] - distance_decelerated)

        # Calculate average velocity and segment time
        average_velocity = (current_velocity + final_velocity) / 2
        segment_times[i] = speed_change_time + remaining_distance / average_velocity

        # Calculate forces
        F_aero = aero_constant * (average_velocity ** 2)
        F_slope = gravity_weight * slopes[i] / 100

        # Calculate energy consumption
        E_drive = (
            (F_rolling + F_aero + F_slope) *
            average_velocity *
            segment_times[i]
        ) / efficiency_factor

        # Add auxiliary power consumption
        E_aux = auxillary_power * 1000 * segment_times[i]

        # Store results
        segment_energies[i] = (E_drive + E_acc + E_aux - E_regen) / 3.6e6
        #energies_regenerated[i] = E_regen / 3.6e6
        final_velocities[i] = final_velocity

    return segment_times, segment_energies, final_velocities


@njit(parallel=True)
def batch_dominates(best_solution, candidates, tolerance=1):
    dominated = np.zeros(len(candidates), dtype=np.bool_)

    # Using prange for parallel execution
    for i in prange(len(candidates)):
        # Extract only first 4 elements and compute difference once
        diff = candidates[i][:4] - best_solution

        # Vectorized operations
        not_worse = np.all(diff >= -tolerance)  # Not worse within tolerance
        strictly_better = np.any(diff > tolerance)  # Strictly better

        # Set result in one operation
        dominated[i] = not_worse and strictly_better

            #candidate = np.int32(candidates[i][4])
            #dominated_nodes.append(candidate)  # Here its the id and NOT the velocity
    #if len(dominated_nodes) > 0:
    #    remove_dominated_paths(dominated_nodes, pareto_front_paths, pareto_front, charging_history, succsessor)
    return dominated


@njit()
def dominates(a0, a1, a2, a3, b0, b1, b2, b3, tolerance=1):
    """Returns:
    - `1` if A dominates B
    - `-1` if B dominates A
    - `0` if non-dominated (Pareto-optimal)
    """

    # Compute differences
    diff0, diff1, diff2, diff3 = b0 - a0, b1 - a1, b2 - a2, b3 - a3

    # Check dominance conditions
    a_dominates = (
        diff0 >= -tolerance and diff1 >= -tolerance and diff2 >= -tolerance and diff3 >= -tolerance and
        (diff0 > tolerance or diff1 > tolerance or diff2 > tolerance or diff3 > tolerance)
    )

    b_dominates = (
        diff0 <= tolerance and diff1 <= tolerance and diff2 <= tolerance and diff3 <= tolerance and
        (diff0 < -tolerance or diff1 < -tolerance or diff2 < -tolerance or diff3 < -tolerance)
    )

    # Return result
    return 1 if a_dominates else -1 if b_dominates else 0


@njit()
def delete_safe_nodes(forced_deletion, pareto_front, pareto_front_paths, charging_history, successor_graph):
    """
    Deletes nodes in batch to reduce dictionary overhead.
    """
    path_ids = List.empty_list(types.int32)

    for node in forced_deletion:
        if node in pareto_front:
            for solution in pareto_front[node]:
                path_ids.append(np.int32(solution[5]))

            pareto_front.pop(node, None)

    for path_id in path_ids:
        pareto_front_paths.pop(path_id, None)
        successor_graph.pop(path_id, None)
        charging_history.pop(path_id, None)


@njit
def remove_dominated_paths(dominated_nodes, pareto_front_paths, charging_history, successor_graph):
    """
    Removes dominated paths and their descendants efficiently.

    Args:
        dominated_nodes (int32): Dominated
        pareto_front_paths (Dict[int32, List[int32]]): Stores paths corresponding to Pareto front nodes.
        successor_graph (Dict[int32, List[int32]]): Tracks successors for efficient deletion.
    """

    while dominated_nodes:
        current = dominated_nodes.pop()  # Get the next node to process

        if current in successor_graph:
            # Process successors
            successors = successor_graph[current]
            for successor in successors:
                if successor in pareto_front_paths:  # Meaning it hasnt already been deleted
                    dominated_nodes.append(successor)
            del pareto_front_paths[current]
            del successor_graph[current]
            del charging_history[current]


@njit()
def paths_are_similar(path1, path2, threshold=0.4):
    """Check if at least `threshold` percent of nodes in path2 exist in path1.
    """

    if len(path2) < 3:
        return False  # Path too short to compare meaningfully

    if path2[-1] in path2[:-1]:  # Uses Python's optimized membership check
        return True

    common_count = 0
    for node2 in path2:
        for node1 in path1:
            if node2 == node1:
                common_count += 1
                break  # Found a match, move to next node2

    # Compute similarity percentage
    similarity = common_count / len(path2)

    return similarity >= threshold




@njit(fastmath=True)
def update_pareto_front(
    pareto_front,      # Dict[int, List[np.ndarray]]
    pareto_front_paths, # Dict[int, List[List[int]]]
    charging_history,
    successors,
    expanded_id,       # int
    new_state,        # np.ndarray
    old_nodes_discovered,  # id of parent
    nodes_discovered,  # int
    dist,            # float
    time,            # float
    energy,          # float
    velocity,        # float
    soc             # float
):
    """
    Update Pareto front with new state.
    Returns: bool - True if front was updated, False if state was dominated
    """

    # Update state
    new_state[0] += dist
    new_state[1] += time
    new_state[2] += energy
    new_state[4] = soc
    new_state[5] = nodes_discovered
    new_state[6] = velocity

    expanded_id_int = np.int32(expanded_id)
    nodes_discovered_int = np.int32(nodes_discovered)
    new_path = typed.List.empty_list(types.int32)
    new_path.extend(pareto_front_paths[old_nodes_discovered])
    new_path.append(expanded_id_int)
    new_charging_history = typed.List.empty_list(types.float32[:])
    new_charging_history.extend(charging_history[old_nodes_discovered])
    existing_solutions = pareto_front.get(expanded_id)
    # If this is a new entry, initialize both dictionaries
    if existing_solutions is None:
        solutions = typed.List.empty_list(types.float32[:])
        solutions.append(new_state[:6])
        pareto_front[expanded_id] = solutions
        pareto_front_paths[nodes_discovered] = new_path
        successors[old_nodes_discovered].append(nodes_discovered_int)
        charging_history[nodes_discovered] = new_charging_history
        return True

    # Get existing solutions and check dominance
    non_dominated = typed.List.empty_list(types.float32[:])
    for existing in existing_solutions:
        existing_parent_id = np.int32(existing[5])
        path = pareto_front_paths.get(existing_parent_id, None)
        if path is None:
            continue

        dominate = dominates(existing[0], existing[1], existing[2], existing[3], new_state[0], new_state[1], new_state[2], new_state[3])

        if dominate == 1:  # new is dominated
            return False
        elif dominate == 0:  # solutions are both optimal
            if abs(existing[4] - new_state[4]) > 0.1:  # Keep both solutions if they have significantly different SOC
                non_dominated.append(existing)
            elif paths_are_similar(path, new_path):  # Check if paths are nearly identical
                return False  # If SOC is similar and paths are similar, discard one
            else:
                non_dominated.append(existing)  # Keep both if paths are different
        elif dominate == -1:  # old entry is dominated, do not add to pareto front again
            dominated_node = typed.List.empty_list(types.int32)
            dominated_node.append(existing_parent_id)
            remove_dominated_paths(dominated_node, pareto_front_paths, charging_history, successors)

    # Add new solution
    non_dominated.append(new_state[:6])
    pareto_front[expanded_id] = non_dominated
    pareto_front_paths[nodes_discovered] = new_path
    successors[old_nodes_discovered].append(nodes_discovered_int)
    charging_history[nodes_discovered] = new_charging_history
    return True


@njit(fastmath=True)
def calculate_consumption(vehicle_data, distance, target_speed, slope, current_velocity):

  weight = vehicle_data[1]
  Cd = vehicle_data[2]
  A = vehicle_data[3]
  F_rolling = vehicle_data[4]
  efficiency_factor = vehicle_data[5]
  regen_efficiency = vehicle_data[6]
  auxillary_power = vehicle_data[7]
  max_acceleration = vehicle_data[8]
  max_deceleration = vehicle_data[9]

  E_acc = 0  # Energy for acceleration
  E_regen = 0  # Energy recovered from braking
  speed_change_time = 0  # Time for acceleration or deceleration
  final_velocity = target_speed  # The speed the car has at the end of the segment
  remaining_distance = distance  # distance remaining after accelerating or braking

  if target_speed > current_velocity:  # Acceleration needed

    # Calculate achievable speed based on available distance
    achievable_speed = np.sqrt(current_velocity**2 + 2 * max_acceleration * distance)

    if achievable_speed < final_velocity:
        final_velocity = achievable_speed

    # Compute acceleration time:
    speed_change_time = (final_velocity - current_velocity) / max_acceleration

    # Energy required for acceleration:
    E_acc = 0.5 * weight * (final_velocity**2 - current_velocity**2)

    # Distance covered during acceleration:
    distance_accelerated = (final_velocity**2 - current_velocity**2) / (2 * max_acceleration)

    # Remaining distance after accelerating
    remaining_distance = distance - distance_accelerated

  elif target_speed < current_velocity:

    # Calculate required braking distance
    required_deceleration_distance = (current_velocity**2 - target_speed**2) / (2 * max_deceleration)

    if required_deceleration_distance > distance:
        # Not enough distance to slow down fully → find the lowest speed achievable
        final_velocity = np.sqrt(current_velocity**2 - 2 * max_deceleration * distance)

    # Deceleration time
    speed_change_time = (current_velocity - final_velocity) / max_deceleration

    # Kinetic energy lost in braking
    E_brake = 0.5 * weight * (current_velocity**2 - final_velocity**2)

    # Regenerated energy (not 100% efficient)
    E_regen = regen_efficiency * E_brake

    # Distance covered during braking
    distance_decelerated = (current_velocity**2 - final_velocity**2) / (2 * max_deceleration)

    remaining_distance = distance - distance_decelerated

  average_velocity = (current_velocity + final_velocity) / 2

  segment_time = speed_change_time + remaining_distance / average_velocity

  # Aerodynamic drag
  F_aero = 0.5 * Cd * A * 1.225 * (average_velocity ** 2) # 1.225 = air density

  # Slope Force (uphill or downhill)
  F_slope = weight * 9.81 * slope / 100

  # Total energy consumption
  E_drive = ((F_rolling + F_aero + F_slope) * average_velocity * segment_time) / efficiency_factor # Divide by efficiency factor (90%)

  # power consumption of lights and electronics
  E_aux = auxillary_power * 1000 * segment_time

  energy = (E_drive + E_acc + E_aux - E_regen) / 3.6e6 # for kWh

  return segment_time, energy, E_regen / 3.6e6, final_velocity
