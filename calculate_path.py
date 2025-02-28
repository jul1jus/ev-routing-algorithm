import numpy as np
import heapq
import time
from scipy.sparse.csgraph import dijkstra
from numba import types, typed
from numba.typed import Dict, List
from compute_path import update_pareto_front, batch_dominates, batch_calculate_consumption, calculate_consumption, distance_to_target, remove_dominated_paths, delete_safe_nodes

# Scale factor for precision (e.g., 0.001 precision)
SCALE_FACTOR = 1
state_dtype = np.float32

def dijkstra_scipy(distance_matrix, source, target):
    """
    Run Dijkstra's algorithm using the optimized adjacency matrix
    Returns the shortest path and the number of nodes discovered.
    """

    # Run Dijkstra using the distance matrix
    distances, predecessors = dijkstra(csgraph=distance_matrix, directed=True, indices=source, return_predecessors=True)
    # Count the number of unique nodes discovered (visited)
    nodes_discovered = np.count_nonzero(predecessors != -9999)

    # If the target is unreachable, return an empty path
    if predecessors[target] == -9999:
        print(f" No path found from {source} to {target}")
        return [], nodes_discovered

    # Reconstruct path
    path = []
    node = target
    while node != -9999:
        path.append(node)
        node = predecessors[node]

    path.reverse()
    print(f"length of path: {len(path)}")
    return path, nodes_discovered


def moop(vehicle_data, nodes, csc_matrix, dist_matrix, speed_matrix, slope_matrix, charging_stations, source_node, target_node, soc):
    """
Input Arguments for moop():

- vehicle_data (np.ndarray): A 1D numpy array with vehicle specifications
- nodes (np.ndarray): A 2D numpy array where each row represents a node with the format [node_id, latitude, longitude].
- dist_matrix (scipy.sparse.csr_matrix): A sparse matrix in CSR format with distances for each edge in the graph.
- speed_matrix (scipy.sparse.csr_matrix): A sparse matrix in CSR format with speed limits for each edge.
- slope_matrix (scipy.sparse.csr_matrix): A sparse matrix in CSR format with slope values for each edge.
- charging_stations (np.ndarray): A structured numpy array of charging stations with fields ['node_id', 'power', 'type'].
- source_node (int): The node ID of the starting point for the routing algorithm.
- target_node (int): The node ID of the destination for the routing algorithm.
"""
    start_time = time.time()
    # cache frequently accessed data
    charging_set = set(charging_stations["node_id"])
    charging_station_dict = {row["node_id"]: row for row in charging_stations}

    target_coords = np.array([nodes[target_node][1], nodes[target_node][2]])
    best_target_path = None

    nodes_discovered = 0  # basically the node id

    # Initialize source node
    source = np.array(
        (0, 0, 0, 0, soc, nodes_discovered, 0),   # distance, time, energy, time_charged, current_soc, nodes_discovered, velocity
        dtype=state_dtype
    )

    queue = [(0, nodes_discovered, source_node, source)]
    heapq.heapify(queue)

    # Store the **best known** NDSP for each node
    pareto_front = Dict.empty(
        key_type=types.int32,
        value_type=types.ListType(types.float32[:])
    )

    pareto_front_paths = Dict.empty(
        key_type=types.int32,
        value_type=types.ListType(types.int32)
    )

    # Track expanded neighbors for each node
    unexpanded_neighbors = {}

    successor_graph = Dict.empty(
        key_type=types.int32,
        value_type=types.ListType(types.int32)
    )

    charging_history = Dict.empty(
        key_type=types.int32,
        value_type=types.ListType(types.float32[:])  # Each entry stores a list of (node, charging_time, SOC)
    )

    # initialize tracking
    unexpanded_neighbors[source_node] = set()

    solutions = typed.List.empty_list(types.float32[:])
    solutions.append(source[:6])
    pareto_front[source_node] = solutions

    # initialize path
    source_path = typed.List.empty_list(types.int32)
    source_path.append(np.int32(source_node))
    source_path.append(np.int32(source_node))  # Add it twice
    pareto_front_paths[nodes_discovered] = source_path
    successor_graph[0] = typed.List.empty_list(types.int32)

    source_parents = set()  # Track parents of source_node

    # Find incoming edges (parents of source_node)
    start_idx, end_idx = csc_matrix.indptr[source_node], csc_matrix.indptr[source_node + 1]
    incoming_parents = csc_matrix.indices[start_idx:end_idx]
    source_parents.update(incoming_parents)

    # initialize charging history
    charging_history[0] = typed.List.empty_list(types.float32[:])

    nodes_discovered += 1

    max_neighbors = 4
    current_batch = []
    deletion_candidates = set()
    nodes_delted = 0

    while queue:
        current_batch.clear()
        queue_len = len(queue)

        if best_target_path is not None:
            batch_size = min(queue_len, 2000)
            current_batch.extend(heapq.heappop(queue) for _ in range(batch_size))
            max_time = 1.5 * best_target_path[1]  # 50% worse time
            max_energy = 1.2 * best_target_path[2]  # 20% worse energy
            filtered_batch = [entry for entry in current_batch if entry[3][1] <= max_time and entry[3][2] <= max_energy]
            if filtered_batch:
                candidates = np.array([
                    [item[0], item[3][1], item[3][2], item[3][3], item[1], item[1]]
                    for item in filtered_batch], dtype=np.float32)
                dominated = batch_dominates(best_target_path[:4], candidates)
                current_batch = [entry for i, entry in enumerate(filtered_batch) if not dominated[i]]
        else:
            batch_size = min(queue_len, 200)
            current_batch.extend(heapq.heappop(queue) for _ in range(batch_size))

        if not current_batch:
            continue

        first_airline = np.int32(current_batch[0][0]-current_batch[0][3][0])
        print(queue_len, batch_size, nodes_discovered, len(pareto_front), first_airline, nodes_delted)

        temp_distances = np.zeros(max_neighbors * len(current_batch), dtype=np.float32)
        temp_speeds = np.zeros_like(temp_distances)
        temp_slopes = np.zeros_like(temp_distances)
        temp_velocities = np.zeros_like(temp_distances)

        total_neighbors = 0
        node_offsets = []  # Track where each node's neighbors start

        # Collect all neighbor data
        for _, _, current_node, current_state in current_batch:
            start_idx = dist_matrix.indptr[current_node]
            end_idx = dist_matrix.indptr[current_node + 1]
            n_neighbors = end_idx - start_idx

            if n_neighbors > 0:
                # Ensure we have enough space
                if total_neighbors + n_neighbors > len(temp_distances):
                    # Resize arrays if needed
                    new_size = max(len(temp_distances) * 2, total_neighbors + n_neighbors)
                    temp_distances = np.resize(temp_distances, new_size)
                    temp_speeds = np.resize(temp_speeds, new_size)
                    temp_slopes = np.resize(temp_slopes, new_size)
                    temp_velocities = np.resize(temp_slopes, new_size)

                # Collect neighbor data
                temp_distances[total_neighbors:total_neighbors + n_neighbors] = \
                    dist_matrix.data[start_idx:end_idx]
                temp_speeds[total_neighbors:total_neighbors + n_neighbors] = \
                    speed_matrix.data[start_idx:end_idx]
                temp_slopes[total_neighbors:total_neighbors + n_neighbors] = \
                    slope_matrix.data[start_idx:end_idx]
                temp_velocities[total_neighbors:total_neighbors + n_neighbors] = current_state[6]  # Store velocity

            node_offsets.append((total_neighbors, n_neighbors))
            total_neighbors += n_neighbors

        # Pass only these extracted arrays to batch_calculate_consumption
        segment_times, segment_energies, new_velocities = batch_calculate_consumption(vehicle_data, temp_distances, temp_speeds, temp_slopes, temp_velocities)

        # Process results for each node
        for batch_idx, (cost, old_nodes_discovered, current_node, current_state) in enumerate(current_batch):
            start_offset, n_neighbors = node_offsets[batch_idx]
            if n_neighbors == 0:
                continue
            parent_path = pareto_front_paths.get(old_nodes_discovered)
            if parent_path is None:
                continue
            parent_id = parent_path[-2]

            parent_neighbors = unexpanded_neighbors.get(parent_id)
            if current_node in parent_neighbors:
                parent_neighbors.discard(current_node)

            unexpanded_neighbors[current_node] = set()

            if old_nodes_discovered not in successor_graph:
                successor_graph[old_nodes_discovered] = typed.List.empty_list(types.int32)

            remaining_kw = current_state[4] * vehicle_data[0]

            # Get neighbors for current node
            neighbors = dist_matrix.indices[
                dist_matrix.indptr[current_node]:
                dist_matrix.indptr[current_node] + n_neighbors
            ]

            # Process feasible neighbors
            feasible_mask = segment_energies[start_offset:start_offset + n_neighbors] <= remaining_kw
            feasible_indices = np.where(feasible_mask)[0]

            new_states = np.zeros((len(feasible_indices), 7), dtype=np.float32)

            for i, idx in enumerate(feasible_indices):
                abs_idx = start_offset + idx
                neighbor = neighbors[idx]

                if nodes_discovered not in successor_graph:
                    successor_graph[nodes_discovered] = typed.List.empty_list(types.int32)

                new_states[i] = current_state.copy()
                current_soc = current_state[4] - (segment_energies[abs_idx] / vehicle_data[0])

                if update_pareto_front(pareto_front, pareto_front_paths, charging_history, successor_graph, neighbor, new_states[i], old_nodes_discovered, nodes_discovered,
                            temp_distances[abs_idx], segment_times[abs_idx], segment_energies[abs_idx], new_velocities[abs_idx], current_soc):
                    if neighbor == target_node:
                        nodes_discovered += 1
                        best_target_path = new_states[i].copy()
                        execution_time = time.time() - start_time
                        print(f"Target node found, {tuple(new_states[i][:4])} nodes_discovered: {nodes_discovered}, batch_size: {batch_size}, time: {execution_time}")
                        continue

                    unexpanded_neighbors[current_node].add(neighbor)
                    distance_left = distance_to_target(nodes[neighbor-1][1], nodes[neighbor-1][2], target_coords[0], target_coords[1])
                    cost = new_states[i][0] + distance_left
                    heapq.heappush(queue, (cost, nodes_discovered, neighbor, new_states[i]))  # Update heap
                    nodes_discovered += 1

                if neighbor in charging_set and current_soc < 0.2:
                    charging_station = charging_station_dict[neighbor]
                    charging_speed = min(charging_station["power"], vehicle_data[-2])  # Type 1 AC
                    if charging_station["type"] == 2:
                        charging_speed = min(charging_station["power"], vehicle_data[-1])
                    charging_time = (1.0 - current_state[4]) * vehicle_data[0] / charging_speed

                    new_state_charged = current_state.copy()
                    new_state_charged[3] += charging_time

                    if update_pareto_front(pareto_front, pareto_front_paths, charging_history, successor_graph, neighbor, new_state_charged, old_nodes_discovered, nodes_discovered,
                            temp_distances[abs_idx], segment_times[abs_idx], segment_energies[abs_idx], new_velocities[abs_idx], 1.0):
                        distance_left = distance_to_target(nodes[neighbor-1][1], nodes[neighbor-1][2], target_coords[0], target_coords[1])
                        cost = distance_left + new_state_charged[0]
                        heapq.heappush(queue, (cost, nodes_discovered, neighbor, new_state_charged))  # Update heap
                        nodes_discovered += 1

        # check if any nodes can be deleted now safely
        nodes_to_delete = List.empty_list(types.int32)
        new_deletion_candidates = set()

        for node in deletion_candidates:
            if node not in pareto_front:
                continue

            if node not in unexpanded_neighbors or unexpanded_neighbors[node]:
                new_deletion_candidates.add(node)
                continue

            # Check if at least one incoming edges (parents) are already deleted
            start_idx, end_idx = csc_matrix.indptr[node], csc_matrix.indptr[node + 1]
            incoming_parents = csc_matrix.indices[start_idx:end_idx]

            all_parents_processed = all(
                parent in unexpanded_neighbors and not unexpanded_neighbors[parent]
                for parent in incoming_parents
            )

            if not all_parents_processed:
                new_deletion_candidates.add(node)
                continue  # Parents or neighbors still exist, not safe to delete

            nodes_to_delete.append(node)

            new_deletion_candidates.update(dist_matrix.indices[dist_matrix.indptr[node]:dist_matrix.indptr[node + 1]])

        if nodes_delted == 0:
            all_parents_processed = all(
                parent in unexpanded_neighbors and not unexpanded_neighbors[parent]
                for parent in source_parents
            )

            if all_parents_processed:
                print(f"Source node {source_node} is now ready for deletion!")
                nodes_to_delete.append(source_node)
                new_deletion_candidates.update(dist_matrix.indices[dist_matrix.indptr[source_node]:dist_matrix.indptr[source_node + 1]])

        if nodes_to_delete:
            delete_safe_nodes(nodes_to_delete, pareto_front, pareto_front_paths, charging_history, successor_graph)
        deletion_candidates = new_deletion_candidates
        nodes_delted += len(nodes_to_delete)

    total_entries = sum(len(pareto_front[key]) for key in pareto_front)
    print(f"pareto_front: {total_entries}, nodes_deleted: {nodes_delted}")

    final_results = []

    for result in pareto_front[target_node]:
        final_results.append((result[0], result[1], result[2], result[3], pareto_front_paths[np.int32(result[5])], charging_history[np.int32(result[5])]))

    return final_results, nodes_discovered


def calculate_consumption_path(vehicle_data, dist_matrix, speed_matrix, slope_matrix, path, charging_stations):
    """
    Computes the total energy consumption, time, and distance for a given path.
    """

    # Find charging stations on the path
    charging_on_path = charging_stations[charging_stations["node_id"].isin(path)]

    total_distance = 0
    total_time = 0.0
    total_energy = 0.0
    total_regenerated_energy = 0.0
    current_velocity = 0.0

    for i in range(len(path) - 1):  # Iterate over edges, not nodes
        start, end = path[i], path[i + 1]

        segment_distance = dist_matrix[start, end]
        segment_speed = speed_matrix[start, end] * (5 / 18)  # Convert km/h to m/s
        segment_slope = slope_matrix[start, end] / 100

        segment_time, segment_energy, energy_regenerated, current_velocity = calculate_consumption(
            vehicle_data, segment_distance, segment_speed, segment_slope, current_velocity
        )

        total_time += segment_time
        total_distance += segment_distance
        total_energy += segment_energy
        total_regenerated_energy += energy_regenerated

    return total_distance, round(total_time / 60, 2), round(total_energy, 3), round(total_regenerated_energy, 3), charging_on_path
