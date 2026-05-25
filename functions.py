import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt

def cut_QTM_min_velocity_start(df, first_samples=50):
    """
    Trim QTM data to start at the frame with minimum velocity in the first X samples.
    First frame zero-velocity issue is handled.
    """
    # 3D positions in cm
    pos = df[['QTM X1', 'QTM Y1', 'QTM Z1']].values / 10.0  
    # Compute velocity magnitude (distance to next frame)
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1)
   
    # Handle case where vel is shorter than first_samples
    search_range = min(first_samples, len(vel))
    if search_range == 0:
        print("Warning: Velocity array is empty. Not trimming.")
        return df
       
    # Ignore first frame being zero; search starting from second frame
    start_idx = np.argmin(vel[:search_range]) + 1  # +1 to match correct frame
    # print(f"Trimming start to frame {start_idx} based on min velocity in first {first_samples} samples.")
   
    # Return trimmed DataFrame
    #print(f"Trimming start to frame {start_idx} based on min velocity.")

    return df.iloc[start_idx:].reset_index(drop=True)


def cut_QTM_min_velocity_start_posZ(df, first_samples=50, trend_length=7):
    """
    Trim QTM data to start at the frame with minimum velocity in the first X samples,
    ensuring that QTM Z shows a rising trend over multiple frames after the minimum.
    If not rising, find the next minimum.
    
    Parameters:
        df: DataFrame with QTM data
        first_samples: Number of initial samples to search for minimum velocity
        trend_length: Number of consecutive frames to check for positive Z trend
    """
    # 3D positions in cm
    pos = df[['QTM X1', 'QTM Y1', 'QTM Z1']].values / 10.0  
    # Compute velocity magnitude (distance to next frame)
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1)
   
    # Handle case where vel is shorter than first_samples
    search_range = min(first_samples, len(vel))
    if search_range == 0:
        print("Warning: Velocity array is empty. Not trimming.")
        return df
    
    # Extract Z positions for checking if it's rising
    z_pos = pos[:, 2]
    
    # Find all local minima in the velocity within first_samples
    local_minima = []
    for i in range(1, min(search_range - 1, len(vel) - 1)):
        if vel[i] < vel[i-1] and vel[i] < vel[i+1]:
            local_minima.append(i)
    
    # If no local minima found, fall back to global minimum in search range
    if not local_minima:
        candidates_idx = [np.argmin(vel[:search_range])]
    else:
        # Sort local minima by velocity value (ascending)
        candidates_idx = sorted(local_minima, key=lambda x: vel[x])
    
    # Try each candidate in order of increasing velocity
    for candidate in candidates_idx:
        start_idx = candidate + 1  # +1 to match correct frame (account for diff reducing length)
        
        # Check if we have enough frames ahead to verify Z trend (need 2x trend_length)
        extended_window = trend_length * 2
        if start_idx >= len(z_pos) - extended_window:
            continue
        
        # Check if Z is consistently rising over trend_length frames
        z_window = z_pos[start_idx:start_idx + trend_length + 1]
        z_diffs = np.diff(z_window)
        
        # Consider it a rising trend if majority of differences are positive
        # and the overall change is positive
        positive_count = np.sum(z_diffs > 0)
        total_change = z_window[-1] - z_window[0]
        
        # Check extended window for velocity reaching at least 0.2
        z_extended = z_pos[start_idx:start_idx + extended_window + 1]
        z_vel_extended = np.diff(z_extended)
        max_z_vel = np.max(z_vel_extended)
        
        # Rising if: majority positive steps AND net positive change AND max Z velocity >= 1
        #positive_count >= trend_length * 0.6
        if total_change > 0 and max_z_vel >= 0.3 and vel[candidate] < 0.5:
            # Found a valid minimum where Z is rising
            return df.iloc[start_idx-1:].reset_index(drop=True)
    
    # If no valid minimum found, use the original minimum (fallback)
    start_idx = np.argmin(vel[:search_range]) + 1
    return df.iloc[start_idx:].reset_index(drop=True)


def get_mean_distances(df,goal='distance',for_each_throw=True):
    distances=[]
    df=df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3']]
    for index, row in df.iterrows():
        
        if goal=='two_point_prec':
            min_distance = float('inf')
            point1, point2 = None, None
            for i in range(0, len(row)-1, 2):
                for j in range(i+2, len(row)-1, 2):
                    x1, y1 = row[i], row[i+1]
                    x2, y2 = row[j], row[j+1]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < min_distance:
                        min_distance = distance
                        point1, point2 = (x1, y1), (x2, y2)
            mean_pointX=(point1[0]+point2[0])/2
            mean_pointY=(point1[1]+point2[1])/2

        else:
            mean_pointX=np.mean (row[[col for col in df.columns if 'x' in col]])
            mean_pointY=np.mean (row[[col for col in df.columns if 'y' in col]])
       
        in_row_distances=[]
        for i in range(0, len(row)-1, 2):
            x1=row[i]
            y1=row[i+1]
            
            if goal=='distance' or goal=='two_point_prec':
                distances.append(np.sqrt((x1-mean_pointX)**2 + (y1-mean_pointY)**2))
            elif goal=='result':
                distances.append(np.sqrt((x1)**2 + (y1)**2))
            elif goal=='x':
                distances.append(np.abs(x1-mean_pointX))
            elif goal=='y':
                distances.append(np.abs(y1-mean_pointY))
            elif goal=='raw_x':
                distances.append(x1)
            elif goal=='raw_y':
                distances.append(y1)
            elif goal=='average':
                in_row_distances.append(np.sqrt((x1-mean_pointX)**2 + (y1-mean_pointY)**2))
            elif goal=='average_x':
                in_row_distances.append(np.abs(x1-mean_pointX))
            elif goal=='average_y':
                in_row_distances.append(np.abs(y1-mean_pointY))
        for i in range(0, len(row)-1, 2):
            if goal=='average' or goal=='average_x' or goal=='average_y':
                distances.append(np.mean(in_row_distances))
    if not for_each_throw:
        distances=distances[0::3]
        

    return distances

def assign_date_integers(df):
    # Initialize an empty dictionary to store unique dates
    unique_dates = {}
    # Counter for assigning integers
    date_counter = 0
    # List to store the integer values corresponding to the dates
    result = []
    
    # Iterate over each row in the 'Time' column
    for time_str in df['time']:
        # Extract the date part (YYYY-MM-DD)
        date_part = time_str.split(" ")[0]  # Or use pd.to_datetime(time_str).strftime('%Y-%m-%d')
        
        # If the date is not already in the dictionary, add it
        if date_part not in unique_dates:
            unique_dates[date_part] = date_counter
            date_counter += 1
        
        # Append the integer corresponding to this date
        for i in range (3): #3 times for each row - because we have 3 throws
            result.append(unique_dates[date_part])
    
    return result

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff=0.3, fs=1.0, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def preprocess(df, f=0.1):
    for column in df.columns:
        df[column] = lowpass_filter(df[column], cutoff=f)
    return df

def create_LA_horizonta_LA_vertical(df):
    for index, row in df.iterrows():
        LAx3=row['LAx3']
        LAy3=row['LAy3']
        LAz3=row['LAz3']
        Grav_vec=np.array([row['GAx'],row['GAy'],row['GAz']])
        LA_hor=np.linalg.norm(np.cross(Grav_vec,[LAx3,LAy3,LAz3]))
        if np.linalg.norm(Grav_vec) == 0:
            print("Gravitational vector is zero, cannot compute vertical component")
            LA_vert = 0  # or handle this case appropriately
            print(Grav_vec)
        else:
            
            LA_vert = np.dot(Grav_vec, [LAx3, LAy3, LAz3]) / np.linalg.norm(Grav_vec)



        df.at[index,'LA_hor']=LA_hor
        df.at[index,'LA_vert']=LA_vert
    return df

def create_zeroed_QTM(dataset,per_throw):
    if per_throw:
        for df in dataset:
            df['QTM X1'] = df['QTM X1'] - df['QTM X1'].iloc[0]
            df['QTM Y1'] = df['QTM Y1'] - df['QTM Y1'].iloc[0]
            df['QTM Z1'] = df['QTM Z1'] - df['QTM Z1'].iloc[0]
    else:
        startX=0
        startY=0
        startZ=0
        for i,df in enumerate(dataset):
            if i%3==0:
                startX=df['QTM X1'].iloc[0]
                startY=df['QTM Y1'].iloc[0]
                startZ=df['QTM Z1'].iloc[0]
            df['QTM X1'] = df['QTM X1'] - startX
            df['QTM Y1'] = df['QTM Y1'] - startY
            df['QTM Z1'] = df['QTM Z1'] - startZ
    return dataset


def remove_bad_columns(loaded_df_list):
    for df_feat in loaded_df_list:
        if 'empty3' in df_feat.columns:
            df_feat.drop(columns=['empty3'], inplace=True)
        if 'LV Ts' in df_feat.columns:
            df_feat.drop(columns=['LV Ts'], inplace=True)
        if 'SAMPLE ID' in df_feat.columns:
            df_feat.drop(columns=['SAMPLE ID'], inplace=True)
        if 'QTM TIME1' in df_feat.columns:
            df_feat.drop(columns=['QTM TIME1'], inplace=True)
        if 'Current event' in df_feat.columns:
            df_feat.drop(columns=['Current event'], inplace=True)
    return loaded_df_list



def generate_features(df_list, feature_set):
    """
    Generate signal features for a list of pandas dataframes and return a single dataframe.
    
    Parameters:
        df_list (list): A list of pandas DataFrames, each containing signal data.
        feature_set (list): A list of strings specifying the features to calculate 
                            (e.g., 'mean', 'std', 'max', 'min', etc.).
    
    Returns:
        pd.DataFrame: A single dataframe containing the calculated features for all input dataframes.
    """
    # Define available feature calculations
    feature_funcs = {
        'mean': lambda x: x.mean(),
        'std': lambda x: x.std(),
        'max': lambda x: x.max(),
        'min': lambda x: x.min(),
        'median': lambda x: x.median(),
        'sum': lambda x: x.sum(),
        'var': lambda x: x.var(),
        'skew': lambda x: x.skew(),
        'kurt': lambda x: x.kurt()
    }
    
    # Check for unsupported features
    unsupported_features = [f for f in feature_set if f not in feature_funcs]
    if unsupported_features:
        raise ValueError(f"Unsupported features in feature_set: {unsupported_features}")
    
    # List to store calculated feature dictionaries
    feature_data = []

    for i, df in enumerate(df_list):
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):  # Process only numeric columns
                for feature in feature_set:
                    feature_data.append({
                        'dataframe': i,  # Add the index of the dataframe
                        'feature': f"{column}_{feature}",
                        'value': feature_funcs[feature](df[column])
                    })

    # Convert the feature data into a single dataframe
    feature_df = pd.DataFrame(feature_data)
    return feature_df.pivot(index='dataframe', columns='feature', values='value').reset_index()


def remove_bad_recordings(df_list, y_list,threshold=4,threshold_duplicates=5):
    #remove recordings,where more than threshold of consequtive samples are identical in LAx3 or QTM X1 columns
    indices_to_remove = []
    y_list = list(y_list)
    for a, df in enumerate(df_list):

        duplicates=0
        for i in range(len(df)-threshold):
            if df['Gx3'].iloc[i] == df['Gx3'].iloc[i+threshold]: #or \
           # df['QTM X1'].iloc[i] == df['QTM X1'].iloc[i+threshold]:
                duplicates+=1
        if duplicates>=threshold_duplicates:      
            indices_to_remove.append(a)
    # Remove items starting from the end
    print(f"Removing {len(indices_to_remove)} recordings")
    print("indices to be removed are ", sorted(indices_to_remove, reverse=True))
    for i in sorted(indices_to_remove, reverse=True):
        print(i)
        df=df_list.pop(i)
        y_list.pop(i)
        print(f"Removed throw number {i} from the recording {i // 3}") 
        
    return df_list, np.array(y_list) , indices_to_remove

def remove_at_index(x_list, y_list, indices):
    # Remove items starting from the end
    for i in sorted(indices, reverse=True):
        x_list.pop(i)
        y_list.pop(i)
        print(f"Removed throw number {i} from the recording {i // 3}")  
    return x_list, y_list