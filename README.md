# Time-Series Discharge Event Detection (Peak → Event Window)

This project detects high-flow discharge events in hourly time series data across multiple gauges.
It identifies peak events and automatically extracts:
- Pre-event start
- Event start
- Event end
- Post-event end

## Why it matters
Automated event-window extraction is useful for flood event analysis, calibration/validation event selection,
and hydrological extreme-event studies.

## Method summary
- Rank-based thresholding (top 1%) per gauge
- Peak detection with `scipy.signal.find_peaks` (height + prominence + minimum distance)
- Rolling-window minima (48h) to estimate event start/end boundaries
- Dynamic event window expansion using discharge-dependent range rules
- Parallel processing across gauges (multiprocessing)

## Outputs
- Excel file with detected events and timestamps
- Event plots (PNG) for each detected event


