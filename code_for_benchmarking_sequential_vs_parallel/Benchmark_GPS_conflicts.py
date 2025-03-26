"""
Vessel Proximity Conflict Detection System

This system identifies potential vessel conflicts by detecting vessels that are too close
to each other at the same timestamp. The implementation includes both sequential and
parallel processing modes for performance benchmarking.

Key Features:
- Haversine distance calculation between vessels
- Time-based grouping of AIS data
- Configurable chunk sizes and worker counts
- Performance benchmarking capabilities

Input: AIS data CSV file with timestamp, MMSI, latitude and longitude columns
Output: CSV file containing detected proximity conflicts and performance metrics
"""

import multiprocessing as mp
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Processing Configuration Constants
SEQ_CHUNK_SIZE = 100000  # Default chunk size for sequential processing
chunk_sizes = [50000, 100000, 200000]  # Chunk sizes to test in parallel processing
number_of_workers = [8, 12, 15]  # Worker counts to test in parallel processing
GROUPING_FREQ = '5min'  # Time window for grouping vessel positions


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two geographic coordinates
    using the Haversine formula.

    Args:
        lat1 (float): Latitude of point 1 in degrees
        lon1 (float): Longitude of point 1 in degrees
        lat2 (float): Latitude of point 2 in degrees
        lon2 (float): Longitude of point 2 in degrees

    Returns:
        float: Distance between points in kilometers
    """
    R = 6371  # Earth radius in km
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Haversine formula components
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def neighbor_vessels(group, conflicting_distance=0.2):
    """
    Identify vessel pairs that are closer than the specified conflict distance.

    Args:
        group (DataFrame): A group of vessel positions within the same time window
        conflicting_distance (float): Distance threshold in km for conflict detection

    Returns:
        list: Dictionary entries for each conflict pair containing:
              - Vessel information for both vessels
              - Calculated distance between them
    """
    anomalies = []
    positions = group[['Latitude', 'Longitude', 'MMSI']].values

    # Compare all unique vessel pairs in the time group
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Skip same vessel comparisons
            if positions[i][2] == positions[j][2]:
                continue

            # Calculate distance between vessels
            dist = calculate_distance(positions[i][0], positions[i][1],
                                      positions[j][0], positions[j][1])

            # Check if distance is below conflict threshold
            if dist < conflicting_distance:
                anomalies.append({
                    "vessel1": {
                        "Timestamp": group.iloc[i]["# Timestamp"],
                        "MMSI": positions[i][2],
                        "Latitude": positions[i][0],
                        "Longitude": positions[i][1]
                    },
                    "vessel2": {
                        "Timestamp": group.iloc[j]["# Timestamp"],
                        "MMSI": positions[j][2],
                        "Latitude": positions[j][0],
                        "Longitude": positions[j][1]
                    },
                    "distance": dist  # Actual calculated distance
                })
    return anomalies


def parallelize_gps(file, chunk_size=100000, num_workers=None):
    """
    Process AIS data in parallel to detect vessel conflicts.

    Args:
        file (str): Path to input CSV file
        chunk_size (int): Number of rows processed per chunk
        num_workers (int): Number of parallel worker processes

    Returns:
        tuple: (list of detected conflicts, processing time in seconds)
    """
    anomalies_gps = []

    # Set default workers if not specified
    if num_workers is None:
        num_workers = mp.cpu_count() - 1

    start_time = time.time()

    # Use context manager for proper pool cleanup
    with mp.Pool(num_workers) as pool:
        chunks = pd.read_csv(file, chunksize=chunk_size)
        total_chunks = sum(1 for _ in pd.read_csv(file, chunksize=chunk_size))

        # Process each chunk with progress bar
        for chunk in tqdm(chunks, total=total_chunks, desc=f'Par Chunk {chunk_size}'):
            # Convert and group timestamps
            chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'],
                                                  format='%d/%m/%Y %H:%M:%S')
            groups = [g for _, g in chunk.groupby(pd.Grouper(key='# Timestamp', freq=GROUPING_FREQ))]

            # Process time groups in parallel
            results = pool.map(neighbor_vessels, groups)
            for res in results:
                anomalies_gps.extend(res)

    end_time = time.time()
    print(f"Parallel {chunk_size}/{num_workers}: {end_time - start_time:.2f}s")
    return anomalies_gps, end_time - start_time


def sequential_gps(file, chunk_size=100000):
    """
    Process AIS data sequentially to detect vessel conflicts.

    Args:
        file (str): Path to input CSV file
        chunk_size (int): Number of rows processed per chunk

    Returns:
        tuple: (list of detected conflicts, processing time in seconds)
    """
    anomalies_gps = []
    start_time = time.time()

    chunks = pd.read_csv(file, chunksize=chunk_size)
    total_chunks = sum(1 for _ in pd.read_csv(file, chunksize=chunk_size))

    # Process each chunk sequentially with progress bar
    for chunk in tqdm(chunks, total=total_chunks, desc='Seq Processing'):
        # Convert and group timestamps
        chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'],
                                              format='%d/%m/%Y %H:%M:%S')
        groups = [g for _, g in chunk.groupby(pd.Grouper(key='# Timestamp', freq=GROUPING_FREQ))]

        # Process each time group sequentially
        for group in groups:
            result = neighbor_vessels(group)
            anomalies_gps.extend(result)

    end_time = time.time()
    print(f"Sequential {chunk_size}: {end_time - start_time:.2f}s")
    return anomalies_gps, end_time - start_time


def run_benchmarks(file_path):
    """
    Execute performance benchmarking across different configurations.

    Args:
        file_path (str): Path to input AIS data file

    Returns:
        DataFrame: Benchmarking results with timing metrics
    """
    print("Starting performance benchmarking...")
    results = []
    #seq_results = sequential_gps(file_path)
    #seq_time = seq_results[2]
    
    # Parallel configuration tests
    for chunk_size in chunk_sizes:
        for workers in number_of_workers:
            try:
                _, par_time = parallelize_gps(file_path, chunk_size, workers)
                results.append({
                    'Chunk_Size': chunk_size,
                    'Workers': workers,
                    'Time(s)': par_time
                })
            except Exception as e:
                print(f"Error in {chunk_size}/{workers}: {str(e)}")

    # Create and display results dataframe
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False, tablefmt="grid"))
    return df


if __name__ == "__main__":
    # Configure working directory and file paths
    os.chdir(r"C:\Users\dania\Big_data")
    CSV_FILENAME = "aisdk-2025-01-30.csv"
    FILE_PATH = os.path.join(os.getcwd(), "Assignment_1", CSV_FILENAME)

    # Validate input file exists
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    # Execute benchmarks and save results
    benchmark_results = run_benchmarks(FILE_PATH)
    benchmark_results.to_csv("gps_conflict_results.csv", index=False)
    print("Results saved to gps_conflict_results.csv")


os.listdrives()