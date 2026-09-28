import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import warnings

def ceilometer_algorithm(ceil_path, met_path, ceil_clr_file, output_csv='output.csv',
                          avg_period=5, date_range=None, wind_thres=3.,
                          vis_thres=10000., rh_thres=90.,
                          window=3, precip_jump_factor=0.25, cat3_thr=10000.):
    
    """
    Modified version of the BLSN detection algorithm of Loeb and Kennedy (2021),
    which identifies blowing-snow events and blowing snow cloud top using ceilometer backscatter profiles
    and basic meteorological thresholds. DOI: 10.1029/2020JD033935

    Modifications relative to the original algorithm:
        - Temporal averaging via optimized xarray resampling.
        - Improved NaN handling and CEIL/MET time alignment.
        - BLSN top detection made less sensitive to noise by using a global 
        peak windowed criterion (see below).

    BLSN top detection:
        The profile is ascended bin by bin from the lowest usable bin. The ascent
        stops when one of the following is met:
        (a) Signal drops below the clear-sky reference → cat 1 (clear-sky BLSN).
        (b) Signal has risen more than precip_jump_factor (25%) above the local
            minimum reached during the ascent → gradual precipitation peak
            detected. The top is assigned to the bin of the local minimum (where
            the profile stopped decreasing) → cat 2 (BLSN + cloud/precip).
        A relative threshold is used in (b) and (c) to ensure robustness across
        events of different intensities. A single-bin noise check (original
        algorithm) is retained for the lowest 10 bins.
        Key parameters: window, precip_jump_factor, cat3_thr (now FUNCTION ARGUMENTS,
        defaults 3, 0.25, 10000. -> identical to the original when left at defaults).

        
    Modules required:
        import xarray as xr
        import numpy as np
        import pandas as pd
        import os
        import glob
        import warnings

    Inputs:
        ceil_path:
            Path to directory containing ceilometer data files
            (supports nested directories; accepts .nc and .cdf ARM/PEA formats).
        met_path:
            Path to directory containing meteorological (MET) data files.
        ceil_clr_file:
            Path to CSV file containing the clear‑sky backscatter profile
            (one single row of values).
        output_csv:
            Name of the CSV file where results are written
            (default = 'output.csv').
        avg_period:
            Time‑averaging period in minutes (default = 5).
        date_range:
            If None: process all available dates.
            If provided: list [start_date, end_date], inclusive of start date,
            exclusive of end date. Format 'YYYY‑MM‑DD'.
        wind_thres:
            Threshold 10 m wind speed (m/s) for BLSN detection (default = 3.0).
        vis_thres:
            Threshold 2 m visibility (m) — included for compatibility but not used
            in the current version (default = 10000.).
        rh_thres:
            Relative humidity threshold (%) for distinguishing fog (default = 90.0).

    Output:
        A pandas DataFrame containing the processed and categorized BLSN events
        for the entire dataset, also saved to `output_csv`.

    Column descriptions:
        'datetime'
            Timestamp marking the start of each averaging period.
        'top_of_detected_BLSN'
            Height (m AGL) of the detected BLSN top; 99999999.9 if not detected.
        'category'
            Event category according to Loeb & Kennedy (2021):
                0 = no BLSN detected
                1 = clear‑sky BLSN
                2 = cloud/precipitation with BLSN
                3 = intense mixed event
                4 = fog
        'lowest_bin_backscatter'
            Backscatter coefficient at the lowest usable bin (10–15 m AGL).
        'decreasing_profile'
            1 if backscatter decreases with height (criterion 1), else 0.
        'above_clear_sky'
            1 if signal at lowest usable bin exceeds clear‑sky profile (criterion 2), else 0.
        '10m_windspeed'
            Averaged 10 m wind speed (m/s).
        '10m_winddirection'
            Averaged 10 m wind direction (deg).
        '2m_temperature'
            Averaged 2 m temperature (°C).
        '2m_rel_hum'
            Averaged 2 m relative humidity (%).
        '2m_visibility'
            Averaged 2 m visibility (m); included for completeness.

    Missing‑value code:
        99999999.9
    """

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    ceil_files = sorted(glob.glob(os.path.join(ceil_path, '**', '*.nc'), recursive=True) +
                        glob.glob(os.path.join(ceil_path, '**', '*.cdf'), recursive=True))
    ceil_files = sorted(list(set(ceil_files)))
    met_files = [f for f in os.listdir(met_path) if os.path.isfile(os.path.join(met_path, f))]
    ceil_clr = pd.read_csv(ceil_clr_file, header=None).values[0]
    
    if os.path.exists(output_csv):
        os.remove(output_csv)

    for file_path in ceil_files:
        try:
            date_str = os.path.basename(file_path).split('.')[-3][-8:]
            if date_range:
                curr = pd.to_datetime(date_str)
                if not (pd.to_datetime(date_range[0]) <= curr < pd.to_datetime(date_range[1])):
                    continue
            
            with xr.open_dataset(file_path) as ds_ceil:
                # Convert decimal hours to datetime
                file_date = pd.to_datetime(date_str)
                time_decimal_hours = ds_ceil['time'].values
                time_datetime = file_date + pd.to_timedelta(time_decimal_hours, unit='h')
                ds_ceil['time'] = time_datetime
                
                bs = ds_ceil.backscatter
                if bs.dims[0] != 'time':
                    bs = bs.transpose('time', 'range')

                bs = bs.assign_coords(time=time_datetime)
                # --- ensure time is monotonic ---
                bs = bs.sortby("time")
                # --- FILTER: remove negative values ---
                # Negative = noise
                bs = bs.where((bs > 0))            
                ds_ceil_res = bs.resample(time=f'{avg_period}min', label='left', closed='left').mean()

            profiles_ceil = ds_ceil_res.values
            avg_times = ds_ceil_res.time.values
            height = ds_ceil['range'].values


            # --- MET LOADING ---
            met_match = [f for f in met_files if date_str in f]
            if not met_match:
                print(f"Met file not found for {date_str}")
                continue
                
            with xr.open_dataset(os.path.join(met_path, met_match[0])) as ds_met:
                ds_met['time'] = pd.to_datetime(ds_met.time.values)
                
                ds_met_res = ds_met[['wspd_arith_mean', 'rh_mean', 'temp_mean', 'wdir_vec_mean', 'pwd_mean_vis_1min']] \
                             .resample(time=f'{avg_period}min', label='left', closed='left').mean() \
                             .reindex(time=avg_times)
                
                wind_avg = ds_met_res.wspd_arith_mean.values
                rh_avg = ds_met_res.rh_mean.values
                temp_avg = ds_met_res.temp_mean.values
                dir_avg = ds_met_res.wdir_vec_mean.values
                vis_avg = ds_met_res.pwd_mean_vis_1min.values

            # --- BLSN DETECTION ALGORITHM ---
            daily_list = []
            for ii in range(len(avg_times)):
                # Initialization
                cat = 0
                blsn_top = 99999999.9
                idx = 0
                
                # NaN handling
                if np.isnan(profiles_ceil[ii, 1]):
                    crit1, crit2 = 0, 0
                else:
                    crit1 = int(profiles_ceil[ii, 1] > np.nanmean(profiles_ceil[ii, 2:7]))
                    crit2 = int(profiles_ceil[ii, 1] > ceil_clr[1])
                
                crit3 = int(wind_avg[ii] > wind_thres) if not np.isnan(wind_avg[ii]) else 0
                crit4 = 1  # visibility criterion deactivated (no visibility data)
                crit5 = int(rh_avg[ii] < rh_thres) if not np.isnan(rh_avg[ii]) else 0

                # BLSN detection - assignment of category and height
                if crit1 + crit2 + crit3 + crit4 + crit5 == 5:
                    idx = 1
                    local_min = profiles_ceil[ii, 1]
                    local_min_idx = 1
                    # window and precip_jump_factor are now function arguments (tunable)

                    while idx < len(height) - 2:

                        # update local minimum if signal is still decreasing
                        if np.isfinite(profiles_ceil[ii, idx]) and profiles_ceil[ii, idx] < local_min:
                            local_min = profiles_ceil[ii, idx]
                            local_min_idx = idx

                        # criterion (a): signal drops below clear-sky reference → BLSN top found
                        if profiles_ceil[ii, idx] < ceil_clr[idx]:
                            break

                        # criterion (b): signal has risen significantly from local minimum
                        if np.isfinite(profiles_ceil[ii, idx]) and local_min > 0:
                            if profiles_ceil[ii, idx] > local_min * (1 + precip_jump_factor):
                                idx = local_min_idx
                                break

                        idx += 1

                    # single-bin noise check for the lowest bins (retained from original algorithm)
                    # if the profile stopped early due to a single noisy bin, step over it and continue
                    if idx < 10 and idx < len(height) - 5:
                        if profiles_ceil[ii, idx] >= np.nanmean(profiles_ceil[ii, idx+2:idx+5]):
                            idx += 1
                            while idx < len(height) - 2:
                                if profiles_ceil[ii, idx] <= profiles_ceil[ii, idx + 1]:
                                    break
                                if profiles_ceil[ii, idx] < ceil_clr[idx]:
                                    break
                                idx += 1
                    
                    # set category
                    if profiles_ceil[ii, idx + 1] < ceil_clr[idx + 1]:
                        cat = 1  # clear sky con BLSN
                    else:
                        cat = 2  # cloud/precip con BLSN
                    
                    if profiles_ceil[ii, 1] >= cat3_thr:
                        cat = 3  # heavy mixed event, regardless of profile shape

                # Check for fog
                if (crit1 + crit2 + crit3 + crit4 == 4) and (crit5 == 0):
                    cat = 4 # BLSN detected but humidity > threshold, therefore category = fog (category 4)
                elif crit1 + crit2 + crit3 + crit4 + crit5 != 5 and cat != 4:
                    cat = 0 # "non-BLSN" profile

                # set BLSN top
                if idx > 0:
                    blsn_top = height[idx]

                daily_list.append([
                    avg_times[ii], blsn_top, cat, 
                    profiles_ceil[ii, 1] if not np.isnan(profiles_ceil[ii, 1]) else 99999999.9,
                    crit1, crit2, 
                    wind_avg[ii] if not np.isnan(wind_avg[ii]) else 99999999.9, 
                    dir_avg[ii] if not np.isnan(dir_avg[ii]) else 99999999.9,
                    temp_avg[ii] if not np.isnan(temp_avg[ii]) else 99999999.9, 
                    rh_avg[ii] if not np.isnan(rh_avg[ii]) else 99999999.9, 
                    vis_avg[ii] if not np.isnan(vis_avg[ii]) else 99999999.9
                ])

            # --- INCREMENTAL SAVE ---
            if daily_list:
                df_daily = pd.DataFrame(daily_list, columns=[
                    'datetime', 'top_of_detected_BLSN', 'category', 'lowest_bin_backscatter',
                    'decreasing_profile', 'above_clear_sky', '10m_windspeed',
                    '10m_winddirection', '2m_temperature', '2m_rel_hum', '2m_visibility'
                ])
                write_header = not os.path.exists(output_csv)
                df_daily.to_csv(output_csv, mode='a', index=False, header=write_header)
            
            print(f"Done: {date_str}")

        except Exception as e:
            print(f"Error on {date_str}: {e}")

    return pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()
