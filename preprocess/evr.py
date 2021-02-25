import os
import sys
import pandas as pd
import datetime
import numpy as np
import math
import re
import utility

def clean_dataframe(df):
    """Return a cleaned dataframe
    
    Clean up the dataframe for or null values (columns) & Sort by Date time
    """
    # Drop columns with NaNs
    df = df.dropna(how='any',axis=1)
    # Sort df based on column 'Date'
    df = df.sort_values(by='Date',ascending=True).reset_index(drop=True)
    # Convert datetime to timestamp (Unix) in milisecond resolution
    df['Timestamp'] = df.Date.values.astype(np.int64) // 10 ** 9 *1000
    # Add column for Real_Time Timestampms
    df['Timestampms'] = df['Timestamp'] 
    
    return df
    
    
def clean_dataframe_test(df, name):
    """Return a cleaned dataframe for test purposes

    Clean up the dataframe for or null values (columns) & Sort by Date time
    """
    # Drop columns with NaNs
    df = df.dropna(how='any',axis=1)
    # Sort df based on column 'Date'
    df = df.sort_values(by='Date',ascending=True).reset_index(drop=True)
    # Convert datetime to timestamp (Unix) in milisecond resolution
    df['Timestamp'] = df.Date.values.astype(np.int64) // 10 ** 9 *1000
    # Add column for Real_Time Timestampms
    df['Timestampms'] = df['Timestamp']

    # Add Prefix to Columns name (e.g. EC2/EOD_***) except Timestampms
    if name == 'C2':
        df.columns = df.columns.map(lambda x : 'EC2_'+x if x !='Timestampms' and x !='Timestamp' else x)
    elif name == 'OD':
        df.columns = df.columns.map(lambda x : 'EOD_'+x if x !='Timestampms' and x !='Timestamp' else x)
        
    # Shift Timestampms to first column
    df = df[ ['Timestampms'] + [ col for col in df.columns if col != 'Timestampms' ] ]

    return df
    
def process_c2_dataframe(df):
    """Return C2 dataframe
    
    Process C2 dataframe to handle duplicates and missing timestamp
    """
    print("---Processing C2 Dataframe---")
    
    df =  clean_dataframe(df)
    # Find duplicates datetime in column 'Timestampms'
    df_duplicate = df[df.duplicated('Timestampms',keep=False)]

    for index, unique_datetime in enumerate (df_duplicate['Timestampms'].unique()):
        timestamp_df = df_duplicate.loc[df_duplicate['Timestampms'] == unique_datetime]
        # print(unique_datetime)
        # Find difference from previous and next timestamp
        difference_previous,  difference_next = find_difference(timestamp_df, unique_datetime, df)
        
        if not(difference_previous == 0 and difference_next == 0) and ((difference_previous <= 1000 and difference_next <= 1000) or
                                                                       (difference_previous == 1000 and difference_next == 0) or
                                                                       (difference_previous == 0 and difference_next == 1000)):
    
            # Split equally e.g 1,2,2,3
            df = split_equally(timestamp_df, df)
        
        elif difference_previous <= 1000 and difference_next >=2000 and  len(timestamp_df.index) < 3:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                df = split_forward(timestamp_df, df)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
        
        elif difference_previous <= 1000 and difference_next >=2000 and  len(timestamp_df.index) > 2:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                # split evenly then fill e.g 6.66, 7, 7, 7, 7, 9 
                df = split_evenly_fill(timestamp_df, df, difference_previous, difference_next)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
            
        elif difference_previous >= 1000 and difference_next <=1000 and  len(timestamp_df.index) > 2:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                # fill then split evenly 
                df = fill_split_evenly(timestamp_df, df, difference_previous, difference_next)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
                
        elif difference_previous == difference_next and difference_previous>1000 and difference_next>1000:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                # fill preceding followed by behind
                df = split_forward_backward(timestamp_df, df)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
                
        elif difference_previous > 1000 and difference_next <=1000 and  len(timestamp_df.index) < 3:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                # Fill Backward e.g 2,4,4,5
                df = split_backward(timestamp_df, df)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
            
        
    # Add Prefix to Columns name (e.g. EC2/EOD_***) except Timestampms
    df.columns = df.columns.map(lambda x : 'EC2_'+x if x !='Timestampms' else x)
        
    # Shift Timestampms to first column
    df = df[ ['Timestampms'] + [ col for col in df.columns if col != 'Timestampms' ] ]
            
    return df

def process_od_dataframe(df, df_c2):
    """Return OD dataframe
    
    Process OD dataframe to handle duplicates and missing timestamp
    """
    print("---Processing OD Dataframe---")
    
    df =  clean_dataframe(df)
    # Find duplicates datetime in column 'Timestampms'
    df_duplicate = df[df.duplicated('Timestampms',keep=False)]
        
    for index, unique_datetime in enumerate (df_duplicate['Timestampms'].unique()):
        timestamp_df = df_duplicate.loc[df_duplicate['Timestampms'] == unique_datetime]
        
        timestamp_df_next = pd.DataFrame()
        if index+1 < len(df_duplicate['Timestampms'].unique()):
            timestamp_df_next = df_duplicate.loc[df_duplicate['Timestampms'] == df_duplicate['Timestampms'].unique()[index+1]]
        timestamp_df_previous = pd.DataFrame()
        if index > 0 :
            timestamp_df_previous = df_duplicate.loc[df_duplicate['Timestampms'] == df_duplicate['Timestampms'].unique()[index-1]]
        
        # print(unique_datetime)
        
        # Check if previous timestamp in C2 has been handled (duplicates, split evenly)
        timestmap_df_c2_previous = df_c2.loc[df_c2['EC2_Timestamp'] == unique_datetime-1000]
        timestmap_df_c2_after = df_c2.loc[df_c2['EC2_Timestamp'] == unique_datetime+1000]
        
        #print(len(timestmap_df_c2_previous), len(timestmap_df_c2_after))
        different_previous_c2 = unique_datetime - timestmap_df_c2_previous.loc[timestmap_df_c2_previous.index[-1], 'Timestampms']
        #print(different_previous_c2)
        timestmap_df_c2 = df_c2.loc[df_c2['EC2_Timestamp'] == unique_datetime]
        
        
        timestamp_to_find = df_c2.loc[timestmap_df_c2.index[-1]-1, 'Timestampms']
        #print (timestamp_to_find, df.loc[timestamp_df.index[0]-1, 'Timestampms'])
     
        # Find difference from previous and next timestamp
        difference_previous,  difference_next = find_difference(timestamp_df, unique_datetime, df)
        
        if not(difference_previous == 0 and difference_next == 0) and ((difference_previous <= 1000 and difference_next <= 1000) or
                                                                       (difference_previous == 1000 and difference_next == 0) or
                                                                       (difference_previous == 0 and difference_next == 1000)):
    
            # Split equally e.g 1,2,2,3
            df = split_equally(timestamp_df, df)
        
        elif difference_previous <= 1000 and difference_next >=2000 and  len(timestamp_df.index) < 3:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                #if len(timestmap_df_c2.index) < len(timestamp_df.index) and different_previous_c2 < 1000:
#                 if len(timestmap_df_c2_previous) != len(timestmap_df_c2_after):
#                     # Split equally e.g 1,2,2,3
#                     df = split_equally(timestamp_df, df)
#                 else:
#                     df = split_forward(timestamp_df, df)
                    
                if timestamp_df_next.index[0]-timestamp_df.index[-1] == 1 \
                    or len(timestmap_df_c2_after)>1 \
                    or len(timestmap_df_c2_previous) == len(timestmap_df_c2_after)\
                    or (df_c2.loc[timestmap_df_c2_previous.index[0],'EC2_Record Number'] < df_c2.loc[timestmap_df_c2_previous.index[-1],'EC2_Record Number'] \
                    and len(timestmap_df_c2_previous) > len(timestamp_df)):
                    df = split_forward(timestamp_df, df)
                else:
                    # Split equally e.g 1,2,2,3
                    df = split_equally(timestamp_df, df)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
        
        elif difference_previous <= 1000 and difference_next >=2000 and  len(timestamp_df.index) > 2:
            
            # split evenly then fill e.g 6.66, 7, 7, 7, 7, 9 
            df = split_evenly_fill(timestamp_df, df, difference_previous, difference_next)
            
        elif difference_previous >= 1000 and difference_next <=1000 and  len(timestamp_df.index) > 2:
            # fill then split evenly 
            df = fill_split_evenly(timestamp_df, df, difference_previous, difference_next)
                
        elif difference_previous == difference_next and difference_previous>1000 and difference_next>1000:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                # fill preceding followed by behind
                df = split_forward_backward(timestamp_df, df)
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
                
        elif difference_previous > 1000 and difference_next <=2000 and  len(timestamp_df.index) < 3:
            if df.loc[timestamp_df.index[0],'Record Number'] < df.loc[timestamp_df.index[-1],'Record Number']:
                #if len(timestmap_df_c2.index) < len(timestamp_df.index):
#                 if len(timestmap_df_c2_previous) != len(timestmap_df_c2_after):
#                     # Split equally e.g 1,2,2,3
#                     df = split_equally(timestamp_df, df)
#                 else:
#                     # Fill Backward e.g 2,4,4,5
#                     df = split_backward(timestamp_df, df)
# 
                #len(timestmap_df_c2_previous)>1    or len(timestmap_df_c2_previous) != len(timestmap_df_c2_after)  len(timestmap_df_c2)>len(timestamp_df) or len(timestmap_df_c2)>1
#                 if (not timestamp_df_previous.empty and timestamp_df.index[0]-timestamp_df_previous.index[-1] == 1) \
#                     or (len(timestmap_df_c2)>1 and len(timestmap_df_c2_previous) != len(timestmap_df_c2_after))\
#                     or len(timestmap_df_c2_previous) == len(timestmap_df_c2_after) \
#                     or len(timestmap_df_c2) == len(timestamp_df):
#                     # Fill Backward e.g 2,4,4,5
#                     df = split_backward(timestamp_df, df)
#                 else:
#                     # Split equally e.g 1,2,2,3
#                     df = split_equally(timestamp_df, df)
                
                timestamp_df_previous_2 = df.loc[df['Timestamp'] == unique_datetime-2000]
                timestamp_df_c2_previous_2 = df_c2.loc[df_c2['EC2_Timestamp'] == unique_datetime-2000]
                # print(len(timestmap_df_c2),len(timestmap_df_c2_previous))    
                
                if len(timestamp_df) > len(timestmap_df_c2) \
                    and len(timestmap_df_c2_previous) >= len(timestmap_df_c2_after) \
                    and len(timestamp_df_c2_previous_2) <= len(timestmap_df_c2_previous) \
                    and len(timestmap_df_c2_previous) > len(timestamp_df) \
                    and len(timestamp_df_c2_previous_2) > len(timestamp_df):
                    #and len(timestamp_df_c2_previous_2) > len(timestamp_df_previous_2):
                    # Split equally e.g 1,2,2,3
                    df = split_equally(timestamp_df, df)
                else:
                    # Fill Backward e.g 2,4,4,5
                    df = split_backward(timestamp_df, df)
                    
                # # if previous datetime has EC2 and EOD
                # if df_c2.loc[timestmap_df_c2_previous.index[0],'EC2_Record Number'] > df_c2.loc[timestmap_df_c2_previous.index[-1],'EC2_Record Number'] \
                    # and len(timestmap_df_c2_previous) < len(timestamp_df):
                    # # Split equally e.g 1,2,2,3
                    # df = split_equally(timestamp_df, df)
                    
                # else:
                    # # Fill Backward e.g 2,4,4,5
                    # df = split_backward(timestamp_df, df)
 
            else:
                # Split equally e.g 1,2,2,3
                df = split_equally(timestamp_df, df)
            
        
    # Add Prefix to Columns name (e.g. EC2/EOD_***) except Timestampms
    df.columns = df.columns.map(lambda x : 'EOD_'+x if x !='Timestampms' else x)
        
    # Shift Timestampms to first column
    df = df[ ['Timestampms'] + [ col for col in df.columns if col != 'Timestampms' ] ]
            
    return df
    
def find_difference(timestamp_df, unique_datetime, df):
    """Return previous and next difference
    
    Calculate time difference for the previous and next timestamp. Use to determine what kind of handling to be applied
    """
    difference_previous, difference_next = 0 , 0
    if timestamp_df.index[0]-1 >=0 and timestamp_df.index[-1]+1 <= len(df)-1:
        difference_previous = unique_datetime - df.loc[timestamp_df.index[0]-1,'Timestampms']
        difference_next = df.loc[timestamp_df.index[-1]+1,'Timestampms'] - unique_datetime
    elif timestamp_df.index[-1] == len(df)-1:
        difference_previous = unique_datetime - df.loc[timestamp_df.index[0]-1,'Timestampms']
    elif timestamp_df.index[0] == 0:
        difference_next = df.loc[timestamp_df.index[-1]+1,'Timestampms'] - unique_datetime
    
    return difference_previous, difference_next

def split_equally(timestamp_df, df):
    """Return dataframe
    
    Based on the number of duplicates, calculate the interval and split the duplicates evenly
    """
    # print("--- Handling Duplicates, Split duplicate, Evenly---")
    # For every Timestamp (1000 millisecond), find the interval
    interval_ms = round(1000 / len(timestamp_df.index))
    for counter in range (0, len(timestamp_df.index)):
        df.loc[timestamp_df.index[counter],'Timestampms'] = df.loc[timestamp_df.index[counter],'Timestampms'] + counter*interval_ms
        # print(df.loc[timestamp_df.index[counter],'Record Number'], df.loc[timestamp_df.index[counter],'Date'],
              # df.loc[timestamp_df.index[counter],'Timestampms'],df.loc[timestamp_df.index[counter],'Record Number'])
    return df


def split_forward(timestamp_df, df):
    """Return dataframe
    
    Based on the number of duplicates, split the duplicates with increasing timestamp - forward
    """
    # print("---Handling Duplicates, Fill gap with duplicate, Forward---")
    for counter in range (0, len(timestamp_df.index)):
        df.loc[timestamp_df.index[counter],'Timestampms'] = df.loc[timestamp_df.index[counter],'Timestampms'] + counter*1000
        # print(df.loc[timestamp_df.index[counter],'Record Number'], df.loc[timestamp_df.index[counter],'Date'],
              # df.loc[timestamp_df.index[counter],'Timestampms'],df.loc[timestamp_df.index[counter],'Record Number'])
    return df

def split_backward(timestamp_df, df):
    """Return dataframe
    
    Based on the number of duplicates, split the duplicates with decreasing timestamp - backward
    """
    # Loop backwards
    # print("---Handling Duplicates, Fill gap with duplicate, Backward---")
    count = 0
    for counter in range (len(timestamp_df.index), 0, -1):
        df.loc[timestamp_df.index[counter-1],'Timestampms'] = df.loc[timestamp_df.index[counter-1],'Timestampms'] - count*1000
        count +=1
        # print(df.loc[timestamp_df.index[counter-1],'Record Number'], df.loc[timestamp_df.index[counter-1],'Date'],
              # df.loc[timestamp_df.index[counter-1],'Timestampms'],df.loc[timestamp_df.index[counter-1],'Record Number'])
    
    return df

def split_forward_backward(timestamp_df, df):
    """Return dataframe

    Based on the number of duplicates, split the duplicates with preceding(first) followed by succeeding missing timestamp
    """
    # print("---Handling Duplicates, Fill gap with duplicate, Forward and Backward---")
    mid_point = len(timestamp_df.index) // 2
    left_count, right_count = 0, 0
    for counter in range (1, len(timestamp_df.index)):
        if counter % 2 == 0:
            right_count +=1
            df.loc[timestamp_df.index[mid_point+right_count],'Timestampms'] = df.loc[timestamp_df.index[mid_point],'Timestampms'] + right_count*1000
            # print(df.loc[timestamp_df.index[mid_point+right_count],'Record Number'], df.loc[timestamp_df.index[mid_point+right_count],'Date'], df.loc[timestamp_df.index[mid_point+right_count],'Timestampms'])
        else:
            left_count +=1
            df.loc[timestamp_df.index[mid_point-left_count],'Timestampms'] = df.loc[timestamp_df.index[mid_point],'Timestampms'] - left_count*1000
            # print(df.loc[timestamp_df.index[mid_point-left_count],'Record Number'], df.loc[timestamp_df.index[mid_point-left_count],'Date'], df.loc[timestamp_df.index[mid_point-left_count],'Timestampms'])
    return df

def split_evenly_fill(timestamp_df, df, difference_previous, difference_next):
    """Return dataframe
    
    Based on the number of duplicates, split the duplicates (part of them) evenly followed by filling the missing timestamp
    """
    # print("---Handling Duplicates, split evenly then fill forward---")
    difference_time_period = int((difference_next - difference_previous) // 1000)
    # print(difference_previous, difference_next, difference_time_period)
    # Split evenly first
    interval_ms = round(1000 / (len(timestamp_df.index)-difference_time_period))
    for counter in range (0, (len(timestamp_df.index)-difference_time_period)):
        df.loc[timestamp_df.index[counter],'Timestampms'] = df.loc[timestamp_df.index[counter],'Timestampms'] + counter*interval_ms
        # print(df.loc[timestamp_df.index[counter],'Record Number'], df.loc[timestamp_df.index[counter],'Date'],
              # df.loc[timestamp_df.index[counter],'Timestampms'],df.loc[timestamp_df.index[counter],'Record Number'])
    # Fill in the gap for forward
    for counter in range ((len(timestamp_df.index)-difference_time_period), len(timestamp_df.index)):
        df.loc[timestamp_df.index[counter],'Timestampms'] = df.loc[timestamp_df.index[counter],'Timestampms'] + 1000
        # print(df.loc[timestamp_df.index[counter],'Record Number'], df.loc[timestamp_df.index[counter],'Date'],
              # df.loc[timestamp_df.index[counter],'Timestampms'],df.loc[timestamp_df.index[counter],'Record Number'])
        
    return df

def fill_split_evenly(timestamp_df, df, difference_previous, difference_next):
    """Return dataframe
    
    Based on the number of duplicates, filling the missing timestamp followed by split the remaining duplicates evenly
    """
    # print("---Handling Duplicates, fill then split evenly---")
    difference_time_period = math.ceil((difference_previous-difference_next) / 1000)
    # print(difference_previous, difference_next, difference_time_period)
    # Fill in the gap for forward
    count = 1
    for counter in range (difference_time_period, 0, -1):
        df.loc[timestamp_df.index[counter-1],'Timestampms'] = df.loc[timestamp_df.index[counter-1],'Timestampms'] - count*1000
        count +=1
        # print(df.loc[timestamp_df.index[counter-1],'Record Number'], df.loc[timestamp_df.index[counter-1],'Date'],
              # df.loc[timestamp_df.index[counter-1],'Timestampms'],df.loc[timestamp_df.index[counter-1],'Record Number'])
    
    # Split evenly 
    interval_ms = round(1000 / (len(timestamp_df.index)-difference_time_period))
    for counter in range (difference_time_period,  len(timestamp_df.index)):
        df.loc[timestamp_df.index[counter],'Timestampms'] = df.loc[timestamp_df.index[counter],'Timestampms'] + (counter-difference_time_period)*interval_ms
        # print(df.loc[timestamp_df.index[counter],'Record Number'], df.loc[timestamp_df.index[counter],'Date'],
              # df.loc[timestamp_df.index[counter],'Timestampms'],df.loc[timestamp_df.index[counter],'Record Number'])
    
    return df
    
def merge_all(df_c2_clean, df_od_clean, start_time, end_time):
    """
    Return meraged dataframe and filepath
    
    Merge all processed C2 and OD dataframes
    """
    print("---Merging all Dataframes (C2 and OD)---")
    # Merge other Operating Data to C2 Dataframe based on Timestampms (ordered manner)
    df_result = pd.merge_ordered(df_c2_clean, df_od_clean, how='outer', on='Timestampms')
    
    #Remove unwanted columns
    df_result.drop(['EC2_Date', 'EC2_Timestamp', 'EOD_Date', 'EOD_Timestamp'], inplace=True, axis=1)
    #Group resulting same real timestamp together
    df_result = df_result.groupby("Timestampms").first().reset_index()
    
    # Rename Real_Timestampms to epoch
    df_result.rename(columns={'Timestampms':'epoch'}, inplace=True)
    
    #df_result.to_csv('./result_11.csv', index=False, header=True)
    
    start_time_new = start_time.strftime('%Y%m%d_%H%M%S')
    end_time_new = end_time.strftime('%Y%m%d_%H%M%S')
    
    filename = "EVR_"+start_time_new+"_to_"+end_time_new+".csv"
    if not os.path.exists('./preprocessing_output'):
        os.makedirs('./preprocessing_output')
    
    filepath = './preprocessing_output/' + filename
    df_result.to_csv(filepath, index=False, header=True)
    
    return df_result, filepath
        
def process_evr (input_folder_path, start_time, end_time):
    """
    Return processed C2, OD, result
    
    Check for folder structure, corresponding EVR files (OD & C2), read logs and filter base on Date
    """
    # Determine time difference between start and end time
    datetime_format = '%Y/%m/%d %H:%M:%S'
    start_time = datetime.datetime.strptime(start_time, datetime_format)
    end_time = datetime.datetime.strptime(end_time, datetime_format)
    start_time_new = start_time.strftime('%m/%d/%Y %H:%M:%S')
    end_time_new = end_time.strftime('%m/%d/%Y %H:%M:%S')
    # Check if EVR folder exist
    if not os.path.exists(os.path.join(input_folder_path, 'EVR')):
        print("EVR folder does not exist. Kindly check the folder strucutre. No EVR processing will be performed")
    else:
        print("EVR folder exist!")
        # Check if C2 file exist
        c2_pattern = 'EVR_Car\d+_\d+_C2.txt'
        norm_pattern = 'EVR_Car\d+_\d+.txt'
        c2_flag = False
        norm_flag = False
        for evr_file in os.listdir(os.path.join(input_folder_path, 'EVR')):
            if re.search(c2_pattern, evr_file):
                c2_filename = evr_file
                c2_flag = True
            if re.search(norm_pattern,evr_file):
                norm_filename = evr_file
                norm_flag = True
            
        if not (c2_flag and norm_flag):
            sys.exit("Missing EVR files! Kindly check that both C2 and Operating Data EVR files are available!")
        else:
            print("---Processing EVR---")
            # Process C2 file
            df_c2 = pd.read_csv(os.path.join(input_folder_path, 'EVR', c2_filename), sep=";")
            df_c2['Date'] = pd.to_datetime(df_c2['Date'], format='%m/%d/%Y %H:%M:%S')
            df_c2_period = df_c2.loc[(df_c2['Date'] >= start_time_new) & (df_c2['Date'] <= end_time_new)]
            df_c2_clean = process_c2_dataframe(df_c2_period)
            #df_c2_clean = process_dataframe(df_c2_period, 'C2')
            #df_c2_clean = clean_dataframe_test(df_c2_period, 'C2')
            
            # Processs Operating Data file
            df_od = pd.read_csv(os.path.join(input_folder_path, 'EVR', norm_filename), sep=";")
            df_od['Date'] = pd.to_datetime(df_od['Date'], format='%m/%d/%Y %H:%M:%S')
            df_od_period = df_od.loc[(df_od['Date'] >= start_time_new) & (df_od['Date'] <= end_time_new)]
            df_od_clean = process_od_dataframe(df_od_period, df_c2_clean)
            #df_od_clean = process_dataframe(df_od_period, 'OD')
            #df_od_clean = clean_dataframe_test(df_od_period, 'OD')
            
            # Merge Processed C2 and OD dataframes
            df_result, filepath = merge_all(df_c2_clean, df_od_clean, start_time, end_time)
                      
            
    return df_c2_clean, df_od_clean, df_result, filepath
    
def output(sample_output_filename, df_result):
    """Output processed result as csv, Testing Purposes
    
    """
    # Import Output File
    df_output = pd.read_csv(sample_output_filename)
    
    df_combine = pd.DataFrame()
    df_combine = df_combine.assign(epoch = df_output['epoch'])
    df_combine = df_combine.assign(Timestampms = df_result['Timestampms']) 
    df_combine = df_combine.assign(EC2_Record_Number = df_result['EC2_Record Number']) 
    df_combine = df_combine.assign(EC2_Date = df_result['EC2_Date']) 
    df_combine = df_combine.assign(EOD_Record_Number = df_result['EOD_Record Number']) 
    df_combine = df_combine.assign(EOD_Date = df_result['EOD_Date']) 

    
    df_combine.to_csv('./result_combine.csv', index=False, header=True)
    
def unit_test(sample_output_filename, name, df_test):
    """
    Perform Unit Test for C2/OD Dataframe
    """
    print("---Unit Test for " + name + " Dataframe---")
    # Unit Test for C2
    # Import Output File
    df_output = pd.read_csv(sample_output_filename)
    # Only retrieve respective columns
    if name == 'C2':
        df_output = df_output.drop(df_output[(df_output['EC2_002_Record_Number'].isnull())].index)
        #drop column with prefix EOD
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('EOD')]
    elif name == 'OD':
        df_output = df_output.drop(df_output[(df_output['EOD_002_Record_Number'].isnull())].index)
        #drop column with prefix EOD
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('EC2')]
    
    df_output['epoch'] = df_output.epoch.values.astype(np.float64)
    df_output = df_output.reset_index(drop=True)
    print(df_output.shape, df_test.shape)    
    #Output to CSV
    #df_output.to_csv('./df_output.csv', index=False, header=True)

    #Output to CSV
    df_test.to_csv('./df_test.csv', index=False, header=True)
    #df_drop_test = pd.read_csv('./df_test.csv')
    df_drop_test = df_test
    df_test['result'] = np.where(df_test['Timestampms'] == df_output['epoch'], 'True', 'False')
    df_test.to_csv('./df_test.csv', index=False, header=True)
    #df_drop_test = df_drop_test.sort_values(by='ATO_Real_Timestampms',ascending=True).reset_index(drop=True)
    # Assert whether sample output and self processed are equal
    assert_equal = utility.nan_equal(df_drop_test['Timestampms'].values, df_output['epoch'].values)
    print("Epoch Equality Between Sample Output and Self Processed: ", assert_equal)
    assert_equal = utility.nan_equal(df_drop_test['E'+name+'_Record Number'].values, df_output['E'+name+'_002_Record_Number'].values)
    print("Record Number Equality Between Sample Output and Self Processed: ", assert_equal)
    
    #df_drop_test.columns = df_output.columns
    #print(np.testing.assert_allclose(df_drop_test.values, df_output.values, rtol=1e-10, atol=0))
    #print(pd.testing.assert_frame_equal(df_drop_test, df_output, check_dtype=False))
    #print(df_drop_test.compare(df_output, align_axis=0))
    #assert_equal = nan_equal(df_drop_test.values, df_output.values)
    #assert_equal = nan_equal(df_drop['ATO_* General'].values, df_output['ATO_0101__General'].values)
    #print("Equality Between Sample Output and Self Processed: ", assert_equal)
    #print(np.testing.assert_equal(df_drop_test.values, df_output.values))
    
def unit_test_all(sample_output_filename, df_test):
    """
    Perform Unit Test for merged (C2 & OD) Dataframe
    """
    print("---Unit Test for entire Dataframe---")
    # Import Output File
    df_output = pd.read_csv(sample_output_filename)
    print(df_output.shape, df_test.shape)
    assert_equal = utility.nan_equal(df_test['epoch'].values, df_output['epoch'].values)
    print("Epoch Equality Between Sample Output and Self Processed: ", assert_equal)
    #df_test['result'] = np.where(df_test['Timestampms'] == df_output['epoch'], 'True', 'False')
    #df_test.to_csv('./df_test.csv', index=False, header=True)
        
    #df_test['result_1'] = np.where(df_test['EC2_Record Number'] == df_output['EC2_002_Record_Number'], 'True', 'False')
    #df_test.to_csv('./df_test.csv', index=False, header=True)
    
    assert_equal = utility.nan_equal(df_test['EC2_Record Number'].values, df_output['EC2_002_Record_Number'].values)
    print("C2 Record Number Equality Between Sample Output and Self Processed: ", assert_equal)
    
    assert_equal = utility.nan_equal(df_test['EOD_Record Number'].values, df_output['EOD_002_Record_Number'].values)
    print("OD Record Number Equality Between Sample Output and Self Processed: ", assert_equal)
    
    #print(np.testing.assert_equal(df_test['EC2_Record Number'].values, df_output['EC2_002_Record_Number'].values))
    #print(np.testing.assert_equal(df_test['EOD_Record Number'].values, df_output['EOD_002_Record_Number'].values))
    
    df_result_2 = df_test.copy(deep=True)
    df_result_2.columns = df_output.columns
    #print(pd.testing.assert_frame_equal(df_result_2, df_output, check_dtype=False, check_column_type=False))
    print("Difference betwen ATLAS Output and Self Generated: ",df_result_2.compare(df_output, align_axis=0))