# GPS_Spoofing_Maritime

# Vessel Anomaly Detection System



## Features

- **Proximity Detection**: Identifies vessels closer than 200m (configurable)
- **Speed Anomalies**: Flags discrepancies between reported and calculated speeds
- **Distance Anomalies**: Detects unrealistic vessel movements between reports
- **Benchmarking**: Compares sequential vs. parallel processing performance
- **Configurable Processing**: Adjustable chunk sizes and worker counts

## System Components

1. **GPS Conflict Detection** (`gps_conflict_detection.py`)
   - Identifies vessels too close to each other
   - Uses Haversine formula for distance calculation
   - Time-window grouping of vessel positions

2. **Speed/Distance Anomaly Detection** (`speed_anomaly_detection.py`)
   - Detects unrealistic speed reports
   - Flags impossible movements between reports
   - Configurable thresholds for anomalies

3. **Benchmarking System**
   - Tests different processing configurations
   - Measures processing times and speedups
   - Generates performance comparison reports