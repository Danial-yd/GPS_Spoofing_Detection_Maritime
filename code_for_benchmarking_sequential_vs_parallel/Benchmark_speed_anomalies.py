"""
Vessel Speed and Distance Anomaly Detection Benchmarking System

This system performs comparative benchmarking between sequential and parallel processing
configurations for detecting anomalies in AIS (Automatic Identification System) data.
The anomalies detected include:
1. Unrealistic vessel movements (distance anomalies)
2. Speed reporting discrepancies (speed anomalies)

The benchmarking tests various chunk sizes and worker counts to identify optimal
processing configurations.
"""

import multiprocessing as mp
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Constants for consistent configuration
SEQ_CHUNK_SIZE = 100000  # Fixed chunk size for sequential processing
chunk_sizes = [50000, 100000, 200000]  # Tested chunk sizes for parallel processing
number_of_workers = [8, 12, 15]  # Tested worker counts for parallel processing


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
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def speed_in(group, max_distance=10, difference_speed=100):
    """
    Detect anomalies in vessel tracking data for a single MMSI group.

    Analyzes consecutive positions of a vessel to identify:
    1. Distance anomalies - unrealistic movements between reports
    2. Speed anomalies - discrepancies between calculated and reported speeds

    Args:
        group (DataFrame): Tracking data for a single vessel (MMSI group)
        max_distance (float): Threshold for distance anomalies in km (default: 10)
        difference_speed (float): Threshold for speed anomalies in km/h (default: 100)

    Returns:
        dict: Dictionary containing two lists:
              - distance_anomalies: Detected distance anomalies
              - speed_anomalies: Detected speed anomalies
    """
    anomalies = []
    speed_anomalies = []
    group = group.sort_values(by="# Timestamp")

    for i in range(1, len(group)):
        # Get consecutive positions
        prev = group.iloc[i - 1]
        current = group.iloc[i]

        # Calculate distance and time difference
        d = calculate_distance(prev['Latitude'], prev['Longitude'],
                               current['Latitude'], current['Longitude'])
        time_diff = (current["# Timestamp"] - prev["# Timestamp"]).total_seconds()

        if time_diff <= 0:
            continue  # Skip invalid time differences

        # Calculate speed from position changes
        calculated_speed = (d / time_diff) * 3600  # km/h
        reported_speed = current['SOG'] * 1.852  # Convert knots to km/h

        # Check distance anomaly
        if d > max_distance:
            anomalies.append({
                "Timestamp": current["# Timestamp"],
                "MMSI": current["MMSI"],
                "Latitude": current["Latitude"],
                "Longtitude": current["Longitude"],  # Note: Original typo preserved
                "SOG": current["SOG"]
            })

        # Check speed discrepancy
        if abs(calculated_speed - reported_speed) > difference_speed:
            speed_anomalies.append({
                "Timestamp": current["# Timestamp"],
                "MMSI": current["MMSI"],
                "Latitude": current["Latitude"],
                "Longtitude": current["Longitude"],
                "SOG": current["SOG"]
            })

    return {"distance_anomalies": anomalies, "speed_anomalies": speed_anomalies}


def parallelize_MMSI(file, chunk_size=100000, num_workers=None):
    """
    Process AIS data in parallel to detect vessel anomalies.

    Args:
        file (str): Path to input CSV file
        chunk_size (int): Number of rows processed per chunk (default: 100000)
        num_workers (int): Number of parallel workers (default: cpu_count-1)

    Returns:
        tuple: (distance_anomalies, speed_anomalies, processing_time)
    """
    distance_anomalies = []
    speed_anomalies = []

    # Set default workers if not specified
    if num_workers is None:
        num_workers = mp.cpu_count() - 1

    pool = mp.Pool(num_workers)

    try:
        start_time = time.time()
        chunks = pd.read_csv(file, chunksize=chunk_size)
        total_chunks = sum(1 for _ in pd.read_csv(file, chunksize=chunk_size))

        for chunk in tqdm(chunks, total=total_chunks, desc=f'Par Chunk {chunk_size}'):
            chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'],
                                                 format='%d/%m/%Y %H:%M:%S')
            groups = [group for _, group in chunk.groupby("MMSI")]
            results = pool.map(speed_in, groups)

            for res in results:
                distance_anomalies.extend(res["distance_anomalies"])
                speed_anomalies.extend(res["speed_anomalies"])

        end_time = time.time()
        print(f"Parallel {chunk_size}/{num_workers}: {end_time - start_time:.2f}s")

    finally:
        pool.close()
        pool.join()

    return distance_anomalies, speed_anomalies, end_time - start_time


def Sequential_MMSI(file, chunk_size=100000):
    """
    Process AIS data sequentially to detect vessel anomalies.

    Args:
        file (str): Path to input CSV file
        chunk_size (int): Number of rows processed per chunk (default: 100000)

    Returns:
        tuple: (distance_anomalies, speed_anomalies, processing_time)
    """
    distance_anomalies = []
    speed_anomalies = []
    start_time = time.time()

    chunks = pd.read_csv(file, chunksize=chunk_size)
    total_chunks = sum(1 for _ in pd.read_csv(file, chunksize=chunk_size))

    for chunk in tqdm(chunks, total=total_chunks, desc='Seq Processing'):
        chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'], format='%d/%m/%Y %H:%M:%S')
        groups = [group for _, group in chunk.groupby("MMSI")]

        for group in groups:
            result = speed_in(group)
            distance_anomalies.extend(result["distance_anomalies"])
            speed_anomalies.extend(result["speed_anomalies"])

    end_time = time.time()
    print(f"\nSequential Baseline: {end_time - start_time:.2f}s")
    return distance_anomalies, speed_anomalies, end_time - start_time


def run_benchmarks(file_path):
    """
    Execute complete benchmarking workflow comparing sequential and parallel processing.

    Args:
        file_path (str): Path to input AIS data file

    Returns:
        DataFrame: Benchmarking results with timing and speedup metrics
    """
    print("Starting performance benchmarking...")

    # Run sequential baseline
    seq_results = Sequential_MMSI(file_path)
    seq_time = seq_results[2]
    # Prepare parallel tests
    results = []

    # Main parallel tests
    print("Running parallel configurations:")
    for chunk_size in chunk_sizes:
        for workers in number_of_workers:
            try:
                _, _, par_time = parallelize_MMSI(file_path, chunk_size, workers)
                results.append({
                    'Chunk_Size': chunk_size,
                    'Workers': workers,
                    'Time(s)': par_time,
                    'Speedup': seq_time / par_time
                })
            except Exception as e:
                print(f" Error in {chunk_size}/{workers}: {str(e)}")

    # Show results
    df = pd.DataFrame(results)
    print("Results:")
    print(df.to_markdown(index=False, tablefmt="grid"))

    return df


if __name__ == "__main__":
    # Configure paths
    os.chdir(r"C:\Users\dania\Big_data")
    CSV_FILENAME = "aisdk-2025-01-30.csv"
    FILE_PATH = os.path.join(os.getcwd(), "Assignment_1", CSV_FILENAME)

    # Validate paths
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    # Execute benchmarks
    benchmark_results = run_benchmarks(FILE_PATH)

    # Optional: Save results to CSV
    benchmark_results.to_csv("performance_results.csv", index=False)
    print("Benchmark results saved to performance_results.csv")