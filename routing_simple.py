import os
os.environ['USE_PYGEOS'] = '0'
import cProfile
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.io import mmread
from scipy.sparse import coo_matrix
import time
import calculate_path
from compute_path import find_closest_node
from shapely.geometry import LineString, Point

def get_max_neighbors_info(csr_graph):
    neighbors_count = np.diff(csr_graph.indptr)
    unique, counts = np.unique(neighbors_count, return_counts=True)
    sorted_indices = np.argsort(unique)[::-1]
    for idx in sorted_indices:
        print(f"Number of neighbors: {unique[idx]}, Count: {counts[idx]}")

def load_matrix(directory_path):
    directory = Path(directory_path)

    # Load adjacency matrix in COO format ALL indicies have shifted +1
    adj_matrix = mmread(directory / "unweighted.mtx").tocoo()

    # Load attributes
    distances = np.loadtxt(directory / "distance.txt", dtype=np.uint16)  # Whole numbers under 10,000
    speeds = (np.loadtxt(directory / "maxspeed.txt", dtype=np.uint8) * (5 / 18)).astype(np.float32)  # Convert to m/s
    slopes = np.loadtxt(directory / "slope.txt", converters={0: lambda x: float(x) / 100}, dtype=np.float32)

    dist_matrix = coo_matrix((distances, (adj_matrix.row, adj_matrix.col)), shape=adj_matrix.shape).tocsr()
    speed_matrix = coo_matrix((speeds, (adj_matrix.row, adj_matrix.col)), shape=adj_matrix.shape).tocsr()
    slope_matrix = coo_matrix((slopes, (adj_matrix.row, adj_matrix.col)), shape=adj_matrix.shape).tocsr()
    csc_matrix = coo_matrix((distances, (adj_matrix.row, adj_matrix.col)), shape=adj_matrix.shape).tocsc()

    # Load node data
    nodes_df = pd.read_csv(directory / "node_mapping.csv", header=None,
                          names=["node_id", "latitude", "longitude"])

    # Convert nodes to numpy array for faster processing
    nodes_array = nodes_df.to_numpy()

    # Load and process charging stations
    charging_stations = pd.read_csv(directory / "charging_stations.csv", sep=";", header=None,
                                  names=["node_id", "Powers", "Type", "point_count", "operator_name"])

    # Process charging station power and type
    charging_stations[['Power', 'Count']] = charging_stations['Powers'].apply(
        lambda x: x.split(',')[0]).str.split(':', expand=True)
    charging_stations = charging_stations.astype({'Power': 'float32', 'Count': 'uint8'})
    charging_stations['Type'] = np.where(charging_stations['Type'].str.strip() == 'CCS', 1, 2)

    # Convert charging stations to structured numpy array
    charging_array = np.zeros(len(charging_stations),
                            dtype=[('node_id', np.int32),
                                  ('power', np.float32),
                                  ('type', np.int32)])

    charging_array['node_id'] = charging_stations['node_id'].to_numpy()
    charging_array['power'] = charging_stations['Power'].to_numpy()
    charging_array['type'] = charging_stations['Type'].to_numpy()

    return csc_matrix, dist_matrix, speed_matrix, slope_matrix, nodes_array, charging_array, charging_stations


def save_path_to_shp(path, charging_stations_used, nodes_array, output_shp="output/shortest_path.shp"):
    """
    Save the path as a shapefile.

    Parameters:
    - path: List of node IDs representing the path.
    - charging_stations_used: DataFrame of charging stations used.
    - nodes_array: Numpy array of shape (N, 3) with columns [node_id, latitude, longitude].
    - output_shp: Path to save the shapefile.
    """

    print(f"Charging Stations found: {len(charging_stations_used)}")

    # Extract coordinates for nodes in the path
    node_ids = nodes_array[:, 0]  # Assuming node_id is in the first column
    path_coords = []
    for node_id in path:
        lon, lat = nodes_array[node_id-1, 2], nodes_array[node_id-1, 1]  # Assuming columns are node_id, latitude, longitude
        path_coords.append((lon, lat))

    path_geometry = LineString(path_coords)
    path_gdf = gpd.GeoDataFrame([{"geometry": path_geometry}], crs="EPSG:4326")
    path_gdf.to_file(output_shp, driver="ESRI Shapefile")

    charging_data = []
    for _, station in charging_stations_used.iterrows():  # iterrows() for DataFrame iteration
        node_id = station['node_id']
        idx = np.where(node_ids == node_id)[0][0]
        lon, lat = nodes_array[idx, 2], nodes_array[idx, 1]
        charging_data.append({
            "geometry": Point(lon, lat),
            "Operator": station['operator_name'],  # Correct for DataFrame
            "kW:count": station['Power'],  # Correct for DataFrame
            "Type": station['Type'],  # Correct for DataFrame
            "Total": station['point_count'],  # Correct for DataFrame
        })

    if len(charging_data) > 0:
        charging_gdf = gpd.GeoDataFrame(charging_data, crs="EPSG:4326")
        charging_gdf.to_file(f"{output_shp}_cs.shp", driver="ESRI Shapefile")

    print(f"Shapefiles saved: {output_shp}")


def main():

  route_count = 1

  vehicle_data = np.array([
        36.6,      # battery_size
        1690,     # weight
        1.756*1.460*0.81*0.28*1.225,  # aero constant A = width * height * 0.81 * Cd = 0.28 * 1.225
        1690* 9.81, # gravity weight
        0.01 * 1690 * 9.81,  # F_rolling
        0.9,      # efficiency_factor
        0.7,      # regen_efficiency
        0.6,       # auxiliary_power
        3,        # max_acceleration
        2,        # max_deceleration
        11000,       # max_ac_power
        70000       # max_dc_power
  ], dtype=np.float32)

  directory_path = "Graph/Dk_De"

  csc_matrix, dist_matrix, speed_matrix, slope_matrix, nodes, charging_array, charging_stations = load_matrix(directory_path)

  starting_soc = 1.0

  while True:
      #get_max_neighbors_info(dist_matrix)
      user_input = input("Source latitude,longitude: ").strip()
      user_input2 = input("Target latitude,longitude: ").strip()
      source_node = find_closest_node(user_input, nodes)
      target_node = find_closest_node(user_input2, nodes)
      algorithm_choice = input("Choose algorithm (1: MOOP, 2: Dijkstra Scipy) ").strip()

      start_time = time.time()

      if algorithm_choice == "1":
        #cProfile.runctx("calculate_path.moop(vehicle_data, nodes, csc_matrix, dist_matrix, speed_matrix, slope_matrix, charging_array, source_node, target_node, starting_soc)",
        #    globals(), locals(), sort="tottime")
        #break
        pareto_front, nodes_discovered = calculate_path.moop(vehicle_data, nodes, csc_matrix, dist_matrix, speed_matrix, slope_matrix, charging_array, source_node, target_node, starting_soc)
        execution_time = time.time() - start_time
        print(f"Paths found in {execution_time:.4f} seconds, with {nodes_discovered} nodes discovered")

        for distance, time_elapsed, energy_consumed, time_charged, path, charging_stations_used in pareto_front:
            print(f"Distance: {distance}m, Time: {round(time_elapsed / 60, 2)}m, Energy: {round(energy_consumed, 3)}kW, total time charging: {time_charged}")
            for charging_station in charging_stations_used:
                print(f"Charged at node {charging_station[0]}, took {charging_station[1]} minutes, began with {charging_station[2]}% SoC")
            charging_station_ids = [entry[0] for entry in charging_stations_used]
            charging_on_path = charging_stations[charging_stations["node_id"].isin(charging_station_ids)]
            save_path_to_shp(path, charging_on_path, nodes, f"output/path{route_count}.shp")
            route_count += 1
      else:
        path, nodes_discovered = calculate_path.dijkstra_scipy(dist_matrix, source_node, target_node)
        execution_time = time.time() - start_time
        print(f"Path found in {execution_time:.4f} seconds, with {nodes_discovered} nodes discovered")
        distance,time_elapsed,energy,energy_regenerated,charging_stations_used = calculate_path.calculate_consumption_path(vehicle_data, dist_matrix, speed_matrix, slope_matrix, path, charging_stations)
        print(f'distance: {distance}m, time: {time_elapsed}m, energy: {energy}kW, energy regenerated: {energy_regenerated}kW')
        save_path_to_shp(path, charging_stations_used, nodes, f"output/path{route_count}.shp")
        route_count += 1


if __name__ == "__main__":
    main()
