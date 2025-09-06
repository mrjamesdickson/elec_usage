#!/usr/bin/env python3
"""
energy_from_cumulative.py

Reads a CSV of cumulative meter readings (kWh) at timestamps and computes:
- HOURLY kWh (splitting each interval across hour boundaries)
- DAILY, WEEKLY, MONTHLY totals (robust resampling from hourly)
- Hourly PROJECTION to the end of the current month using a rolling-mean run-rate
  (enabled by default; disable with --no-project)
- Optionally include projected hours in the rollups with --agg-include-projection

Expected columns by default:
  - time        (timestamp)
  - meter_kwh   (cumulative kWh reading from the meter)

Examples:
  python energy_from_cumulative.py readings.csv --tz America/New_York
  python energy_from_cumulative.py readings.csv --tz America/New_York --projection-mean-hours 6
  python energy_from_cumulative.py readings.csv --no-project
  python energy_from_cumulative.py readings.csv --time-col Timestamp --kwh-col Cumulative -o usage_summary.csv
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# ---------------------------- CLI ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute hourly/daily/weekly/monthly kWh from cumulative meter readings, with month-end hourly projection via rolling mean."
    )
    p.add_argument("csv", help="Path to input CSV file.")
    p.add_argument("--time-col", default="time", help="Timestamp column name (default: time).")
    p.add_argument("--kwh-col", default="meter_kwh", help="Cumulative kWh column name (default: meter_kwh).")
    p.add_argument("--tz", default=None,
                   help="IANA timezone for naive timestamps (e.g., 'America/New_York'). "
                        "If provided, naive times are localized; if timestamps already have tz, they are converted.")
    p.add_argument("--week-anchor", default="SUN",
                   choices=["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"],
                   help="Week ending day for weekly totals (default: SUN).")
    p.add_argument("--no-reset", action="store_true",
                   help="Disable reset handling. If a reading decreases, that interval is dropped instead of treated as a reset to ~0.")
    # Projection controls (defaults ON with rolling mean)
    p.add_argument("--no-project", action="store_true",
                   help="Disable the end-of-month hourly projection.")
    p.add_argument("--projection-mean-hours", type=int, default=24,
                   help="Rolling window (hours) for projection run-rate (default: 24).")
    p.add_argument("--agg-include-projection", action="store_true",
                   help="Include projected hourly values in daily/weekly/monthly aggregates.")
    p.add_argument("-o", "--output", default=None,
                   help="Optional CSV path for tidy combined output (hourly/daily/weekly/monthly + projected_hourly if enabled).")
    p.add_argument("--graphs", action="store_true", default=True,
                   help="Generate and save usage graphs (default: True, use --no-graphs to disable).")
    p.add_argument("--no-graphs", action="store_true",
                   help="Disable graph generation.")
    p.add_argument("--graph-dir", default="./output",
                   help="Directory to save graph images (default: ./output, use empty string for screen display).")
    return p.parse_args()


# ------------------------- Utilities -------------------------

def ensure_timezone(dt_index: pd.DatetimeIndex, tz: Optional[str]) -> pd.DatetimeIndex:
    if tz is None:
        return dt_index
    if dt_index.tz is None:
        return dt_index.tz_localize(tz)
    return dt_index.tz_convert(tz)


def week_freq(week_anchor: str) -> str:
    return f"W-{week_anchor.upper()}"


def try_alias_kwh_column(df: pd.DataFrame, current: str) -> str:
    """If expected kWh column is missing, try common aliases (case-sensitive first, then case-insensitive)."""
    if current in df.columns:
        return current
    # direct common names
    for cand in ["kwh", "cumulative", "total_kwh", "reading_kwh", "energy", "meter", "meter reading", "meter_kW h"]:
        if cand in df.columns:
            return cand
    # case-insensitive fallback
    lower = {str(c).lower(): c for c in df.columns}
    for cand in ["meter_kwh", "kwh", "cumulative", "total_kwh", "reading_kwh", "energy", "meter"]:
        if cand in lower:
            return lower[cand]
    return current  # let downstream dropna raise a clear error


@dataclass
class CumInterval:
    start: pd.Timestamp
    end: pd.Timestamp
    kwh: float  # energy used during the interval


def build_intervals_cumulative(df: pd.DataFrame, tcol: str, cumcol: str, allow_reset: bool) -> List[CumInterval]:
    """
    Build intervals [t[i-1], t[i]) with energy = diff of cumulative readings.
    If allow_reset and a reset/rollover is detected (diff < 0), treat interval kWh = reading[i] (assume reset to ~0).
    """
    out: List[CumInterval] = []
    n = len(df)
    for i in range(1, n):
        t0 = pd.Timestamp(df.at[i - 1, tcol])
        t1 = pd.Timestamp(df.at[i, tcol])
        v0 = df.at[i - 1, cumcol]
        v1 = df.at[i, cumcol]

        if pd.isna(t0) or pd.isna(t1) or pd.isna(v0) or pd.isna(v1):
            continue
        if t1 <= t0:
            continue

        delta = float(v1) - float(v0)
        if delta < 0:
            if allow_reset:
                kwh = float(v1)  # reset: count new reading as usage since reset
            else:
                continue
        else:
            kwh = delta

        if kwh < 0:
            continue

        out.append(CumInterval(start=t0, end=t1, kwh=kwh))
    return out


def _next_hour_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    base = ts.floor("H")
    return base + pd.Timedelta(hours=1)


def split_interval_by_hour(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    segs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    s = start
    while s < end:
        e = min(end, _next_hour_boundary(s))
        segs.append((s, e))
        s = e
    return segs


def allocate_interval_kwh_by_time(intervals: List[CumInterval]) -> Dict[pd.Timestamp, float]:
    """
    Split each interval's kWh proportionally by time across hour buckets.
    Returns mapping from hour_start -> kWh.
    """
    hourly: Dict[pd.Timestamp, float] = {}
    for iv in intervals:
        total_hours = pd.Timedelta(iv.end - iv.start).total_seconds() / 3600.0
        if total_hours <= 0:
            continue
        segs = split_interval_by_hour(iv.start, iv.end)
        for (cs, ce) in segs:
            hours = pd.Timedelta(ce - cs).total_seconds() / 3600.0
            if hours <= 0:
                continue
            share = iv.kwh * (hours / total_hours)
            key = cs.floor("H")
            hourly[key] = hourly.get(key, 0.0) + share
    return hourly


def summarize(hourly_df: pd.DataFrame, week_anchor: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Robust rollups from hourly -> daily/weekly/monthly.
    Assumes hourly_df index is a sorted DatetimeIndex and column 'kWh' is numeric.
    """
    # Ensure numeric and sorted
    s = pd.to_numeric(hourly_df["kWh"], errors="coerce").dropna().sort_index()

    # Daily totals (calendar day)
    daily = s.resample("D").sum(min_count=1).to_frame(name="kWh")
    daily.index.name = "date"

    # Weekly totals (periods ending on chosen anchor day)
    wf = week_freq(week_anchor)
    weekly = s.resample(wf, label="right", closed="right").sum(min_count=1).to_frame(name="kWh")
    weekly.index.name = f"week_end_{week_anchor.upper()}"

    # Monthly totals (calendar month end)
    monthly = s.resample("M").sum(min_count=1).to_frame(name="kWh")
    monthly.index.name = "month_end"

    return daily, weekly, monthly


def rolling_mean_runrate(hourly_df: pd.DataFrame, window_hours: int) -> Optional[float]:
    """
    Rolling-mean run-rate (kWh/hour) over the last `window_hours`.
    If fewer rows than window, use all available. Returns None if empty.
    """
    if hourly_df.empty:
        return None
    window = max(1, int(window_hours))
    tail = pd.to_numeric(hourly_df["kWh"], errors="coerce").dropna().tail(window)
    if tail.empty:
        return None
    return float(tail.mean())


def project_to_end_of_month(hourly_df: pd.DataFrame, window_hours: int) -> Optional[pd.DataFrame]:
    """
    Projection: repeat the rolling-mean hourly kWh (over the last `window_hours` hours)
    for each hour from the next hour after the last observed hour until the final
    hour of the current month (23:00 on the last day).
    """
    if hourly_df.empty:
        return None

    last_ts = hourly_df.index.max()
    runrate = rolling_mean_runrate(hourly_df, window_hours)
    if runrate is None:
        return None

    # Start projecting from the next full hour after the last observed hour
    start_future = last_ts.floor("H") + pd.Timedelta(hours=1)

    # Last hour start of the month (23:00 at last calendar day)
    month_end_day_midnight = (last_ts + pd.offsets.MonthEnd(0)).normalize()
    final_hour_start = month_end_day_midnight + pd.Timedelta(hours=23)

    if start_future > final_hour_start:
        return None

    total_hours = int(pd.Timedelta(final_hour_start - start_future).total_seconds() // 3600) + 1
    idx = pd.date_range(start=start_future, periods=total_hours, freq="H")
    proj = pd.DataFrame({"kWh": [runrate] * total_hours}, index=idx)
    proj.index.name = "hour_start"
    return proj


def create_graphs(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, weekly_df: pd.DataFrame, 
                 monthly_df: pd.DataFrame, proj_month_df: Optional[pd.DataFrame], 
                 graph_dir: Optional[str], week_anchor: str) -> None:
    """Generate and display/save usage graphs."""
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Hourly usage time series (top half)
    ax1 = plt.subplot(2, 2, (1, 2))
    hourly_df.plot(kind='line', ax=ax1, color='steelblue', linewidth=1.5)
    
    # Add projection if available
    if proj_month_df is not None:
        proj_month_df.plot(kind='line', ax=ax1, color='orange', linestyle='--', linewidth=2, alpha=0.8)
        ax1.legend(['Actual Usage', 'Projected Usage'])
    else:
        ax1.legend(['Usage'])
    
    ax1.set_title('Hourly Electricity Usage', fontsize=14, fontweight='bold')
    ax1.set_ylabel('kWh')
    ax1.grid(True, alpha=0.3)
    
    # Use automatic date locator to avoid tick overflow
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=20))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # 2. Daily usage bar chart (bottom left)
    ax2 = plt.subplot(2, 2, 3)
    daily_df.plot(kind='bar', ax=ax2, color='lightcoral', width=0.8)
    ax2.set_title('Daily Usage', fontsize=12, fontweight='bold')
    ax2.set_ylabel('kWh')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend().set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Weekly/Monthly comparison (bottom right)
    ax3 = plt.subplot(2, 2, 4)
    
    # Combine weekly and monthly data for comparison
    if not weekly_df.empty and not monthly_df.empty:
        # Create a combined plot with different bar widths
        x_pos = range(len(monthly_df))
        monthly_values = monthly_df['kWh'].values
        
        bars1 = ax3.bar([x - 0.2 for x in x_pos], monthly_values, 0.4, 
                       label='Monthly', color='darkgreen', alpha=0.7)
        
        # Show most recent weeks
        recent_weeks = weekly_df.tail(min(len(monthly_df) * 4, len(weekly_df)))
        if not recent_weeks.empty:
            week_x_pos = range(len(recent_weeks))
            bars2 = ax3.bar([x + 0.2 for x in week_x_pos[:len(monthly_df)]], 
                           recent_weeks['kWh'].tail(len(monthly_df)), 0.4,
                           label=f'Weekly (end {week_anchor})', color='mediumseagreen', alpha=0.7)
        
        ax3.set_title('Monthly vs Weekly Usage', fontsize=12, fontweight='bold')
        ax3.set_ylabel('kWh')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis with month names
        month_labels = [ts.strftime('%b %Y') for ts in monthly_df.index]
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(month_labels, rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for\nmonthly/weekly comparison', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Monthly vs Weekly Usage', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    
    # Save graphs as PNG
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(graph_dir) / 'electricity_usage_graphs.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraphs saved to: {filepath}")
    plt.close()


# --------------------------- Main ----------------------------

def main():
    args = parse_args()

    # Read CSV
    df = pd.read_csv(
        args.csv,
        parse_dates=[args.time_col],
        keep_default_na=True,
    )

    # Try to resolve cumulative column if user didn't pass correct name
    args.kwh_col = try_alias_kwh_column(df, args.kwh_col)

    # Clean & sort
    if args.time_col not in df.columns or args.kwh_col not in df.columns:
        raise SystemExit(f"Missing expected columns. Found: {list(df.columns)}. "
                         f"Pass --time-col / --kwh-col to match your CSV headers.")
    df = df.dropna(subset=[args.time_col, args.kwh_col]).sort_values(by=args.time_col).reset_index(drop=True)
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df[args.kwh_col] = pd.to_numeric(df[args.kwh_col], errors="coerce")
    df = df.dropna(subset=[args.time_col, args.kwh_col])

    # Timezone handling
    tindex = pd.DatetimeIndex(df[args.time_col])
    tindex = ensure_timezone(tindex, args.tz)
    df[args.time_col] = tindex

    # Remove exact duplicate timestamps
    df = df[~df[args.time_col].duplicated(keep="first")].reset_index(drop=True)

    # Build intervals from cumulative readings
    intervals = build_intervals_cumulative(df, args.time_col, args.kwh_col, allow_reset=not args.no_reset)
    if not intervals:
        raise SystemExit("No valid intervals found (need at least two rows with increasing timestamps).")

    # Allocate to hourly buckets
    hourly_map = allocate_interval_kwh_by_time(intervals)
    if not hourly_map:
        raise SystemExit("No hourly energy could be allocated (check data).")

    hourly_df = pd.DataFrame({"kWh": pd.Series(hourly_map)}).sort_index()
    hourly_df.index.name = "hour_start"

    # Ensure clean index and numeric values before any resampling
    hourly_df = hourly_df[~hourly_df.index.isna()].sort_index()
    hourly_df["kWh"] = pd.to_numeric(hourly_df["kWh"], errors="coerce")
    hourly_df = hourly_df.dropna(subset=["kWh"])

    # Projection (rolling-mean run-rate to month end) - enabled by default
    proj_month_df = None
    if not args.no_project:
        proj_month_df = project_to_end_of_month(hourly_df, args.projection_mean_hours)

    # Choose aggregation base: actuals only, or include projection
    agg_base = hourly_df
    if args.agg_include_projection and proj_month_df is not None:
        agg_base = pd.concat([hourly_df, proj_month_df]).sort_index()

    # Summaries
    daily_df, weekly_df, monthly_df = summarize(agg_base, args.week_anchor)

    # Pretty print
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    print("\n=== HOURLY USAGE (kWh) ===")
    print(hourly_df.to_string())

    if proj_month_df is not None:
        print(f"\n=== PROJECTED HOURLY to MONTH END (rolling mean over last {args.projection_mean_hours} hours) ===")
        print(proj_month_df.to_string())

    print("\n=== DAILY USAGE (kWh) ===")
    print(daily_df.to_string())

    print(f"\n=== WEEKLY USAGE (kWh) [weeks end on {args.week_anchor.upper()}] ===")
    print(weekly_df.to_string())

    print("\n=== MONTHLY USAGE (kWh) ===")
    print(monthly_df.to_string())

    # Optional tidy CSV
    if args.output:
        hourly_out  = hourly_df.reset_index().assign(level="hourly").rename(columns={"hour_start": "period_start"})
        daily_out   = daily_df.reset_index().assign(level="daily").rename(columns={"date": "period_end"})
        weekly_out  = weekly_df.reset_index().assign(level="weekly").rename(columns={weekly_df.index.name: "period_end"})
        monthly_out = monthly_df.reset_index().assign(level="monthly").rename(columns={monthly_df.index.name: "period_end"})

        frames = [
            hourly_out[["level", "period_start", "kWh"]],
            daily_out[["level", "period_end", "kWh"]],
            weekly_out[["level", "period_end", "kWh"]],
            monthly_out[["level", "period_end", "kWh"]],
        ]

        if proj_month_df is not None:
            proj_out = proj_month_df.reset_index().assign(level="projected_hourly").rename(columns={"hour_start": "period_start"})
            frames.append(proj_out[["level", "period_start", "kWh"]])

        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nWrote summaries to: {args.output}")

    # Generate graphs by default unless disabled
    if args.graphs and not args.no_graphs:
        graph_dir = args.graph_dir if args.graph_dir else None
        create_graphs(hourly_df, daily_df, weekly_df, monthly_df, proj_month_df, graph_dir, args.week_anchor)


if __name__ == "__main__":
    main()
