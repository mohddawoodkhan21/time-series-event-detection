import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import os

# Function to read and preprocess time series data
def read_time_series(file_path, delimiter):
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        df = df.rename(columns={'Unnamed: 0': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Function to find minima within a rolling window of 48 hours
def find_minima_rolling(discharge_df, column, peak_idx, window_size=48, direction="backward"):
    total_length = len(discharge_df)
    
    if direction == "backward":
        start = max(0, peak_idx - window_size)
        if start == peak_idx:
            print(f"Peak is too close to the start of data for column {column}. Skipping.")
            return None
        rolling_min = discharge_df[column].iloc[start:peak_idx].rolling(window=window_size, min_periods=1).min()
    else:
        end = min(total_length - 1, peak_idx + window_size)
        if end == peak_idx:
            print(f"Peak is too close to the end of data for column {column}. Skipping.")
            return None
        rolling_min = discharge_df[column].iloc[peak_idx:end].rolling(window=window_size, min_periods=1).min()

    min_idx = rolling_min.idxmin()

    # Validate the minimum index
    if pd.isna(min_idx) or min_idx is None:
        print(f"Rolling window returned invalid minima index in column {column}. Skipping event.")
        return None

    return discharge_df.index.get_loc(min_idx)  # Convert index to integer

# Function to analyze a single gauge (column)
def analyze_gauge(args):
    column, discharge_df, temp_df, precip_df, output_dir = args
    results = []

    # Calculate rank-based threshold for peak detection
    discharge_df['Rank'] = discharge_df[column].rank(method='first', ascending=False)
    top_1_rank = discharge_df['Rank'].quantile(0.01)
    discharge_1st_threshold = discharge_df.loc[discharge_df['Rank'] <= top_1_rank, column].min()

    # Calculate the average of the top 1% discharge values for the current gauge
    top_1_percent_values = discharge_df.loc[discharge_df['Rank'] <= top_1_rank, column]
    threshold_for_gauge = top_1_percent_values.mean()  # Average threshold for this gauge

    # Detect peaks using Prominence Filtering and minimum distance between peaks
    peaks, _ = find_peaks(discharge_df[column], height=discharge_1st_threshold, prominence=9, distance=460)

    for i, peak in enumerate(peaks):
        peak_time = discharge_df.index[peak]
        peak_discharge = discharge_df[column].iloc[peak]

        # Skip peaks with discharge below the threshold calculated for this gauge
        if peak_discharge < threshold_for_gauge:
            print(f"Skipping peak at {peak_time} for {column} as peak_discharge ({peak_discharge}) < threshold ({threshold_for_gauge})")
            continue
        
        # Determine the dynamic max range based on the peak discharge
        if peak_discharge <= 100:
            max_range = 0.05 * peak_discharge  # 5% for discharges between 0 and 100
        else:
            max_range = 0.10 * peak_discharge  # 10% for discharges greater than 100

        ### Step 1: Find the start event time using a rolling window for minima detection
        start_idx = find_minima_rolling(discharge_df, column, peak, window_size=48, direction="backward")
        if start_idx is None or start_idx < 0 or start_idx >= len(discharge_df):
            print(f"Invalid start index for {column} at peak {peak_time}. Skipping this event.")
            continue  # Skip if no valid start index found
        event_start_time = discharge_df.index[start_idx]

        ### Step 2: Find the end event time using a rolling window for minima detection
        end_idx = find_minima_rolling(discharge_df, column, peak, window_size=48, direction="forward")
        if end_idx is None or end_idx < 0 or end_idx >= len(discharge_df):
            print(f"Invalid end index for {column} at peak {peak_time}. Skipping this event.")
            continue  # Skip if no valid end index found
        event_end_time = discharge_df.index[end_idx]

        ### Step 3: Find pre-event and post-event times based on the max_range
        pre_event_window_start = max(0, start_idx - 48)  # Ensure no out-of-bound indexing
        pre_event_window = discharge_df.loc[discharge_df.index[pre_event_window_start]:event_start_time, column]
        within_range_pre = (pre_event_window >= 0) & (pre_event_window <= max_range)
        pre_event_start_time = pre_event_window[within_range_pre].index[0] if within_range_pre.any() else event_start_time

        post_event_window_end = min(len(discharge_df) - 1, end_idx + 48)  # Ensure no out-of-bound indexing
        post_event_window = discharge_df.loc[event_end_time:discharge_df.index[post_event_window_end], column]
        within_range_post = (post_event_window >= 0) & (post_event_window <= max_range)
        post_event_end_time = post_event_window[within_range_post].index[-1] if within_range_post.any() else event_end_time

        # Handle exceptional cases where pre-event/start-event or end-event/post-event are the same
        if (pre_event_start_time == event_start_time or event_end_time == post_event_end_time):
            print(f"Pre-event or post-event overlap detected for {column} at {peak_time}. Adjusting max_range.")

            # Adjust max_range to 10% of peak_discharge
            max_range = 0.10 * peak_discharge

            # Recalculate pre-event and post-event times
            within_range_pre = (pre_event_window >= 0) & (pre_event_window <= max_range)
            pre_event_start_time = pre_event_window[within_range_pre].index[0] if within_range_pre.any() else event_start_time

            within_range_post = (post_event_window >= 0) & (post_event_window <= max_range)
            post_event_end_time = post_event_window[within_range_post].index[-1] if within_range_post.any() else event_end_time

            # If still the same, increase by 5 units and check again
            if pre_event_start_time == event_start_time or post_event_end_time == event_end_time:
                print(f"Max_range adjustment failed for {column} at {peak_time}. Adding 5 units to max_range.")
                max_range += 5

                # Recalculate pre-event and post-event times
                within_range_pre = (pre_event_window >= 0) & (pre_event_window <= max_range)
                pre_event_start_time = pre_event_window[within_range_pre].index[0] if within_range_pre.any() else event_start_time

                within_range_post = (post_event_window >= 0) & (post_event_window <= max_range)
                post_event_end_time = post_event_window[within_range_post].index[-1] if within_range_post.any() else event_end_time

        # Append results
        results.append({
            'Column': column,
            'Peak Time': peak_time,
            'Pre-Event Start Time': pre_event_start_time,
            'Event Start Time': event_start_time,
            'Event End Time': event_end_time,
            'Post-Event End Time': post_event_end_time,
            'Peak Discharge': peak_discharge
        })

        # Create focused dataframe for plotting (including all points from Pre-Event Start Time to Post-Event End Time)
        focused_df = discharge_df.loc[pre_event_start_time:post_event_end_time, column]

        # Check for NaN values in the time window between Pre-Event Start and Post-Event End
        if focused_df.isna().any():
            print(f"Skipping plot for {column} at {peak_time} due to NaN values within the event.")
            continue

        # Plotting the event
        plt.figure(figsize=(10, 6))
        plt.plot(focused_df.index, focused_df, label='Discharge', color='blue')
        plt.axvline(x=pre_event_start_time, color='green', linestyle='--', label='Pre-Event Start')
        plt.axvline(x=event_start_time, color='orange', linestyle='--', label='Event Start')
        plt.axvline(x=peak_time, color='red', linestyle='-', label='Peak Time')
        plt.axvline(x=event_end_time, color='orange', linestyle='--', label='Event End')
        plt.axvline(x=post_event_end_time, color='green', linestyle='--', label='Post-Event End')
        plt.scatter([peak_time], [peak_discharge], color='red', label='Peak Discharge', zorder=5)
        plt.fill_between(focused_df.index, 0, focused_df, color='blue', alpha=0.3)
        plt.title(f'Discharge Event - Gauge {column} - Peak at {peak_time}')
        plt.xlabel('Date')
        plt.ylabel('Discharge')
        plt.legend()

        # Save the plot
        plot_filename = f'{column}_event_{peak_time.strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()

    # Check if results list is empty before converting to DataFrame
    if len(results) > 0:
        result_df = pd.DataFrame(results)
        result_df['Rank'] = result_df['Peak Discharge'].rank(method='first', ascending=False)
        return result_df.to_dict('records')
    else:
        print(f"No valid events found for column {column}.")
        return []

# Main script
if __name__ == "__main__":
    discharge_file_path =Put file path
    temperature_file_path = Put file path
    precipitation_file_path = Put file path
    output_dir = Put file path
    os.makedirs(output_dir, exist_ok=True)

    discharge_df = read_time_series(discharge_file_path, delimiter=';')
    temperature_df = read_time_series(temperature_file_path, delimiter=';')
    precipitation_df = read_time_series(precipitation_file_path, delimiter=';')

    if discharge_df is not None and temperature_df is not None and precipitation_df is not None:
        common_columns = discharge_df.columns.intersection(temperature_df.columns).intersection(precipitation_df.columns).tolist()
        common_index = discharge_df.index.intersection(temperature_df.index).intersection(precipitation_df.index)
        discharge_df = discharge_df.loc[common_index]
        temperature_df = temperature_df.loc[common_index]
        precipitation_df = precipitation_df.loc[common_index]

        temperature_filter = (temperature_df[common_columns] >= 5).all(axis=1)
        discharge_df = discharge_df[temperature_filter]
        temperature_df = temperature_df[temperature_filter]
        precipitation_df = precipitation_df[temperature_filter]

        args = [(column, discharge_df, temperature_df, precipitation_df, output_dir) for column in common_columns]

        with Pool(cpu_count()) as pool:
            results = pool.map(analyze_gauge, args)

        results = [item for sublist in results for item in sublist]

        if len(results) > 0:
            top_discharge_events_df = pd.DataFrame(results)
            output_file_path = Put file path
            try:
                top_discharge_events_df.to_excel(output_file_path, index=False)
                print(f"Top discharge events data has been written to {output_file_path}")
            except Exception as e:
                print(f"Error writing to Excel file {output_file_path}: {e}")
        else:
            print("No valid events found.")
    else:
        print("Failed to read and process the discharge, temperature, or precipitation data.")

