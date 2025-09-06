# Electricity Usage Analysis

A Python tool for analyzing cumulative electricity meter readings and generating comprehensive usage reports with visualization.

## Features

- **Hourly Analysis**: Converts cumulative meter readings to hourly usage, splitting intervals across hour boundaries
- **Multi-Period Summaries**: Generates daily, weekly, and monthly usage totals
- **Usage Projection**: Estimates remaining monthly usage using rolling average of recent consumption
- **Visual Analytics**: Automatic generation of usage graphs including time series and comparison charts
- **Flexible Input**: Handles various CSV formats with configurable column names and timezones
- **Meter Reset Handling**: Intelligently detects and handles meter resets/rollovers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mrjamesdickson/elec_usage.git
cd elec_usage
```

2. Create and activate virtual environment:
```bash
python3 -m venv env
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python elec_usage.py readings.csv
```

### With Timezone
```bash
python elec_usage.py readings.csv --tz America/New_York
```

### Custom Column Names
```bash
python elec_usage.py readings.csv --time-col Timestamp --kwh-col Cumulative
```

### Disable Graphs
```bash
python elec_usage.py readings.csv --no-graphs
```

### Quick Start
Use the included shell script for automated setup and execution:
```bash
./run.sh
```

## Input Format

Expected CSV format:
```csv
time,meter_kwh
2024-01-01 00:00:00,1000.5
2024-01-01 01:00:00,1002.3
2024-01-01 02:00:00,1004.1
```

The tool automatically detects common column name variations:
- Time columns: `time`, `timestamp`, `date`, etc.
- kWh columns: `meter_kwh`, `kwh`, `cumulative`, `reading_kwh`, etc.

## Output

### Console Output
- Hourly usage table
- Projected hourly usage (if enabled)
- Daily, weekly, and monthly summaries

### Graph Output
Automatically saves graphs to `./output/electricity_usage_graphs.png`:
- Hourly usage time series with projection overlay
- Daily usage bar chart
- Monthly vs weekly usage comparison

### CSV Export
Optional tidy CSV output with `--output filename.csv` containing all time periods in a normalized format.

## Command Line Options

| Option | Description |
|--------|-------------|
| `--time-col` | Timestamp column name (default: time) |
| `--kwh-col` | Cumulative kWh column name (default: meter_kwh) |
| `--tz` | IANA timezone (e.g., 'America/New_York') |
| `--week-anchor` | Week ending day for weekly totals (default: SUN) |
| `--no-reset` | Disable meter reset handling |
| `--no-project` | Disable end-of-month projection |
| `--projection-mean-hours` | Rolling window for projection (default: 24) |
| `--agg-include-projection` | Include projected values in summaries |
| `--no-graphs` | Disable graph generation |
| `--graph-dir` | Directory for graph output (default: ./output) |
| `-o, --output` | CSV path for tidy output |

## Examples

### Analyze with projection and save all outputs:
```bash
python elec_usage.py data.csv --tz America/New_York --projection-mean-hours 12 -o summary.csv
```

### Weekly analysis ending on Monday:
```bash
python elec_usage.py data.csv --week-anchor MON
```

### Include projections in aggregates:
```bash
python elec_usage.py data.csv --agg-include-projection
```

## Requirements

- Python 3.6+
- pandas >= 1.3.0
- matplotlib >= 3.3.0

## License

MIT License