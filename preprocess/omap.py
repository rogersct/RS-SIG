import os
import sys
import pandas as pd
import datetime
import numpy as np
import time
import re
from tqdm import tqdm
tqdm.pandas()
import timeit
import collections
import itertools
import utility
import fnmatch

def clean_dataframe(df):
    """Return a dataframe
    
    Clean up the dataframe for any duplicates or null values
    """
    count_start = len(df)
    
    # Does it make sense to drop duplicates?
#     # Clean - Drop duplicates entries
#     df.drop_duplicates(keep=False,inplace=True)
#     # Clean - Check for duplicates counts
#     print("Number of duplicates dropped = ",  count_start-len(df))

    # Clean - Drop null value entries
    count_start = len(df)
    df = df.dropna(how='any',axis=0)
    # Clean - Check for null value counts
    print("Number of null value entry = ",  count_start-len(df))
    return df

def validate_datetime(datetime_str):
    format_str = '%m/%d/%Y'
    try:
      datetime.datetime.strptime(datetime_str, format_str)
      return True
    except ValueError:
      return False
     
  
def clean_datetime(df):
    df['Date'] = df['Date'].apply(lambda x: x+" 00:00:00" if validate_datetime(x) else x)
    #print(df.head(20))
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    return df

def distribute_equal (interval_list):
    """Return a list
        
        Re-order the list and distribute the intervals equally
    """
    # Get count of each unique interval
    unique_value_collection = collections.Counter(interval_list)
    unique_value = unique_value_collection.keys()
    unique_value_count = unique_value_collection.values()
    unique_value, unique_value_count = list(unique_value), list(unique_value_count)
    #print(unique_value, unique_value_count)

    remain_list = unique_value.copy()
    min_index = unique_value_count.index(min(unique_value_count))
    max_index = unique_value_count.index(max(unique_value_count))
    remain_list.remove(unique_value[min_index])

    master_list = []
    counter = 0
    # Generate sets based on the min count e.g. [20,19]
    for x in range (0, min(unique_value_count)):
        unique_value_copy = unique_value.copy()
        for y in remain_list:
            #print(unique_value_collection[y])
            #print(unique_value_collection[y]//min(unique_value_count))
            # If there are more counts of the other interval (aside of the min interval), we'll need to append to the base set
            if(unique_value_collection[y]//min(unique_value_count)!=1):
                for z in range (0, unique_value_collection[y]//min(unique_value_count)-1):
                    unique_value_copy.append(y)
        counter = counter + len(unique_value_copy)
        master_list.append(unique_value_copy)

    # Flattened_list
    master_list = list(itertools.chain.from_iterable(master_list))
    
    # Append the remaining un-arranged interval based on it's value (e.g. is it bigger or smaller than min interval)
    if counter != len(interval_list):
        if unique_value[max_index] > unique_value[min_index]:
            master_list.insert(0, unique_value[max_index])
        else:
            master_list.append(unique_value[max_index])
    print(master_list)
    return master_list
    
def check_for_monotonic_old (df):
    """ Return a dataframe
    
    Detect and clean unsequence real time

    """
    if not df['Realtime'].is_monotonic_increasing:
        # Get index of non-monotonic location - non increasing order
        df_non_monotonic = df.loc[df['Realtime'].diff() < pd.to_timedelta('0 seconds')]
        
        # *** Need to determine if for loop forward or backwards (changing to which real time index) - Not Implemented
        
        print("Found non-monotonic sequence at index: ", (df.loc[df['Realtime'].diff() < pd.to_timedelta('0 seconds')].index))
        for index in df_non_monotonic.index:
            non_monotonic_realtime = df['Realtime'][index-1]
            count_non_monotonic_realtime = len(df[df['Realtime'] == non_monotonic_realtime].index)
                       
            # Find start location of index with the same real time (index)
            start_index = min(df[df['Realtime'] == df['Realtime'][index]].index)
            count_first_realtime = len(df[df['Realtime'] == df['Realtime'][index]].index)
            
            total_count = count_first_realtime + count_non_monotonic_realtime
            windows = total_count // 2
            first_realtime_windows = total_count - windows
            
            # Next time index information
            start_index_next_realtime = start_index + first_realtime_windows
                               
            # Update First Real Time
            for count in range (0,first_realtime_windows):
                df.loc[start_index+count,'Realtime'] = df['Realtime'][index]
                print(start_index+count, df.loc[start_index+count,'Realtime'])
                                
            # Update Real time for the next time index
            for count in range (0,windows):
                df.loc[start_index_next_realtime+count,'Realtime'] = non_monotonic_realtime
                print(start_index_next_realtime+count, df.loc[start_index_next_realtime+count,'Realtime'])
    return df
    
def check_for_monotonic (df):
    """ Return a dataframe
    
    Detect and clean unsequence real time

    """
    if not df['Realtime'].is_monotonic_increasing:
        # Get index of non-monotonic location - non increasing order
        df_non_monotonic = df.loc[df['Realtime'].diff() < pd.to_timedelta('0 seconds')]
        
        # *** Need to determine if for loop forward or backwards (changing to which real time index) - Not Implemented
        
        print("Found non-monotonic sequence at index: ", (df.loc[df['Realtime'].diff() < pd.to_timedelta('0 seconds')].index))
        for index in df_non_monotonic.index:
            non_monotonic_realtime = df['Realtime'][index-1]
            #Difference in real time in seconds 
            difference_realtime = (df['Realtime'][index-1]-df['Realtime'][index]).total_seconds()
            
            affected_realtime_list = []
            total_count_non_monotonic_sequences = 0
            for value in range (0, int(difference_realtime) +1):
                affected_realtime = df['Realtime'][index] + datetime.timedelta(seconds=value)
                # Find total count of affected sequences
                total_count_non_monotonic_sequences =  total_count_non_monotonic_sequences + len(df[df['Realtime'] == affected_realtime].index)
                #print(affected_realtime, len(df[df['Realtime'] == affected_realtime].index))
                affected_realtime_list.append(affected_realtime)
            
            print(affected_realtime_list)
             
            # Find start index of the first timestamp (index)
            start_index = min(df[df['Realtime'] == df['Realtime'][index]].index)
            
            # Calculate interval
            interval_list = []
            for value in range (0, int(difference_realtime) +1):
                intervals = total_count_non_monotonic_sequences // (int(difference_realtime) +1 - value)
                total_count_non_monotonic_sequences = total_count_non_monotonic_sequences - intervals
                interval_list.append(intervals)
            
            # Reverse list
            interval_list.reverse()
            print(interval_list)
            
            # If the intervals are not the same, it needs special handling
            if (interval_list.count(interval_list[0]) != len(interval_list)):
                interval_list = distribute_equal(interval_list)

            continue_index = 0
            # Update Real time          
            for x in range (0, len(interval_list)):
                # Update Real Time
                for count in range (0,interval_list[x]):
                    df.loc[start_index + continue_index + count,'Realtime'] = affected_realtime_list[x]
                    #print(start_index + continue_index + count, df.loc[start_index + continue_index + count,'Realtime'])
            
                continue_index = continue_index + interval_list[x]
            # # If divisible by length of affected realtime
            # if total_count_non_monotonic_sequences % (int(difference_realtime) +1) == 0:
                # windows = int(total_count_non_monotonic_sequences / (int(difference_realtime) +1))
                # for value in range (0, len(affected_realtime_list)):
                    # # Update Real Time
                    # for count in range (0,windows):
                        # df.loc[start_index + windows*value + count,'Realtime'] = affected_realtime_list[value]
                        # #df.loc[start_index+count,'Realtime'] = df['Realtime'][index]
                        # print(start_index + windows*value + count, df.loc[start_index + windows*value + count,'Realtime'])
    return df

def preprocess_machine_time(df):
    """Return a dataframe
    
    Convert Machine Time (Column 'Date') to Unix Timestamp with Millisecond Resolution
    """
    # Date to Unix Timestamp
    #df.info(verbose=True)
    # Convert Date object (mm/dd/yyyy hh:mm:ss)to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')

    # Convert datetime to timestamp (Unix) in milisecond resolution
    df['Timestamp'] = df.Date.values.astype(np.int64) // 10 ** 9 *1000

    #df['Converted Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    #df.head(20)
    #df.info(verbose=True)

    # Add column for Timestampms
    df['Timestampms'] = ""
    df['Timestampms'].replace('','0',inplace=True)
    df["Timestampms"] = df["Timestampms"].astype(np.int64)
    
#     # Add column for interval_ms
#     df['interval_ms'] = ""
#     df['interval_ms'].replace('','0',inplace=True)
#     df["interval_ms"] = df["interval_ms"].astype(np.int64)
#     #df.info(verbose=True)

    # # Create empty dataframe
    # df2 = pd.DataFrame(data=None, columns=df.columns)

    # list_df = []
    # list_df.append(df2)
    #Iterate through unique Timestamp - Add milisecond to Machine Time
    for item in sorted(df["Timestamp"].unique()):
        #timestamp_df = df.query('Timestamp==@item')
        timestamp_df = df.loc[df['Timestamp'] == item]
        # For every Timestamp (1000 millisecond), find the interval
        interval_ms = round(1000 / timestamp_df.shape[0])
        counter = 0
        for index, row in timestamp_df.iterrows():
            #print(index)
            df.loc[index,'Timestampms'] = row.Timestamp + counter*interval_ms
            #df.loc[index,'interval_ms'] = interval_ms
            counter += 1
    
    #list_df.append(timestamp_df)
    return df

            
def generate_real_time(x):
    """Return real time
    
    Check if real time needs to increase by 1 day
    """
    time_difference = datetime.timedelta(hours = x.Hour, minutes = x.Minute, seconds = x.Second) - datetime.timedelta(hours = real_time_start.hour, minutes = real_time_start.minute, seconds = real_time_start.second)
    datetime_real_time = datetime.datetime.combine(machine_date_start, datetime.time(x.Hour, x.Minute, x.Second))
                                                  
    if(time_difference < datetime.timedelta(days=0)):
        #print("Increase by 1 day")
        datetime_real_time = datetime_real_time + datetime.timedelta(days=1)

    x['Realtime'] = datetime_real_time
    return x['Realtime']
                                                   
    
def preprocess_real_time(df, back_date_flag):
    """Return a dataframe
    
    Convert Real Time (Columns 'Hour', 'Minute', Second) to datetime.time with Millisecond Resolution
    """
    # Combine Hours Minutes and Seconds to datetime
    #df['Millisecond'] = ""
    df['Realtime'] = ""
    df['Realtime'].replace('','0',inplace=True)
    df["Realtime"] = df["Realtime"].astype(np.int64)
    global real_time_start 
    real_time_start = datetime.time(df.Hour[0], df.Minute[0], df.Second[0])
    global machine_date_start    
    machine_date_start = datetime.date(df.Date[0].year, df.Date[0].month, df.Date[0].day)
    if back_date_flag:
        machine_date_start = machine_date_start - datetime.timedelta(days=1) 
        print(machine_date_start)
    
    df['Realtime'] = df.apply(generate_real_time, axis=1)                                               
#     df['Realtime'] = df.apply(lambda row: 
#                               datetime.datetime.combine(row.Date.date(), datetime.time(row.Hour, row.Minute, row.Second)), 
#                               axis=1)
    # Check for unsequence Real Time (Non-Monotonic)
    df = check_for_monotonic(df)
    
    # Convert Date object (mm/dd/yyyy hh:mm:ss)to datetime type
    df['Realtime'] = pd.to_datetime(df['Realtime'], format='%m/%d/%Y %H:%M:%S')

    # Convert datetime to timestamp (Unix) in milisecond resolution
    df['Real_Timestamp'] = df.Realtime.values.astype(np.int64) // 10 ** 9 *1000
    
    #df.info(verbose=True)
    
    # Add column for Real_Time Timestampms
    df['Real_Timestampms'] = ""
    df['Real_Timestampms'].replace('','0',inplace=True)
    df["Real_Timestampms"] = df["Real_Timestampms"].astype(np.int64)

    #Iterate through unique Real_Timestamp - Add milisecond to Real Time
    for item in sorted(df["Real_Timestamp"].unique()):
        #timestamp_df = df.query('Timestamp==@item')
        real_timestamp_df = df.loc[df['Real_Timestamp'] == item]
        # For every Timestamp (1000 millisecond), find the interval
        interval_ms = round(1000 / real_timestamp_df.shape[0])
        counter = 0
        for index, row in real_timestamp_df.iterrows():
            #print(index)
            df.loc[index,'Real_Timestampms'] = row.Real_Timestamp + counter*interval_ms
            counter += 1
    
    return df
    
def nearest(lst, K):
    """Return a index for the closest value
    
    Find the closest value inside the list
    """
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K)) 

def calculate_ATO_real_timestampms(x, df_result):
    """Return ATO Real Time and nearest index
    
    With the unmapped ATP machine time, we need to calculate it's equivalent ATO Real Time by
    finding the closest ATO machine time
    """
    if(pd.isnull(x['Real_Timestampms'])):
        if x.name == 0:
            # The closest is the next index
            x['nearest_index'] = 1
            x['Real_Timestampms'] = abs(df_result.loc[x.name+1,'Timestampms'] - df_result.loc[x.name,'Timestampms']) + df_result.loc[x.name+1,'Real_Timestampms']
        elif x.name < df_result.shape[0]-1:    
            # Find the closest between the index before and after
            difference_before = df_result.loc[x.name-1,'Timestampms']
            difference_after = df_result.loc[x.name+1,'Timestampms']
            index_timestampms = df_result.loc[x.name,'Timestampms']
            difference_list = [difference_before, difference_after]
            # Find nearest index
            nearest_index = nearest(difference_list, index_timestampms)
            x['nearest_index'] = nearest_index
            if nearest_index == 0:
                # Compute ATO Real Time - if nearest_index is the index before, take difference between the two Machine time
                # and add it to the equivalent ATO real time to preseve the ATP's time interval
                x['Real_Timestampms'] = abs(index_timestampms - difference_before) + df_result.loc[x.name-1,'Real_Timestampms']
            else:
                # Compute ATO Real Time - if nearest_index is the index after, take difference between the two Machine time
                # and add it to the equivalent ATO real time to preseve the ATP's time interval
                x['Real_Timestampms'] = abs(difference_after - index_timestampms) + df_result.loc[x.name+1,'Real_Timestampms']
        else:
            # The closest is the previous index
            x['nearest_index'] = 0
            x['Real_Timestampms'] = abs(df_result.loc[x.name-1,'Timestampms'] - df_result.loc[x.name,'Timestampms']) + df_result.loc[x.name-1,'Real_Timestampms']
            
    
    return pd.Series([x['Real_Timestampms'], x['nearest_index']])

def indicate_switch(x, df_result):
    """Return nearest index
    
    With previous row's nearest_index indicating 1 (needs to be switched after the index after),
    set the current row's nearest_index as 2
    """
    if(x.name>0):
        if df_result.loc[x.name-1,'nearest_index'] == 1:
            x['nearest_index'] = 2
    
    return x['nearest_index']

def switch_row_position(x, df_result):
    """Return dataframe
    
    Retrieve row based on nearest_index to perform switching of row when required
    """
    if(x['nearest_index']==1):
        #if nearest_index is the index after, switch
        return df_result.iloc[x.name+1]
    elif(x['nearest_index']==2):
        return df_result.iloc[x.name-1]
    else:
        #if nearest_index is the index before, remain the order
        return df_result.iloc[x.name]
        
def process_ato(df, back_date_flag):
    """Return a dataframe
    
    Process ATO dataframe
    """
    print("---Processing ATO Dataframe---")
    #Clean ATO dataframe
    df = clean_dataframe(df)

    # Preprocess Machine Time
    df = preprocess_machine_time(df)
    #df2 = pd.concat(list_df) # Concat a list of dataframes is faster than appending individual dataframes
    # Preprocess Real Time
    df = preprocess_real_time(df, back_date_flag)
    
    print(df.loc[df['Realtime'].diff() < pd.to_timedelta('0 seconds')].index)
    #df.to_csv('./clean_real_time.csv', index=False, header=True)
    # Drop Date, Timestamp & Real Time Timestamp Columns
    df.drop(['Date', 'Timestamp', 'Realtime','Real_Timestamp'], inplace=True, axis=1)

    # Add Prefix to Columns name (ATO_***) except Timestampms
    df.columns = df.columns.map(lambda x : 'ATO_'+x if x !='Timestampms' and x!='Real_Timestampms' else x)
    # Shift Timestampms to first column
    df = df[ ['Real_Timestampms'] + [ col for col in df.columns if col != 'Real_Timestampms' ] ]
    df = df[ ['Timestampms'] + [ col for col in df.columns if col != 'Timestampms' ] ]
    
    return df


def process_others(df, name):
    """Return a dataframe
    
    Process ATP/COM/TDMS dataframe
    """
    print("---Processing "+ name+ " Dataframe---")
    #Clean dataframe
    df = clean_dataframe(df)
    # Preprocess Machine Time
    df = preprocess_machine_time(df)
    df.drop(['Date', 'Timestamp'], inplace=True, axis=1)
    # Add Prefix to Columns name (TDMS_***) except Timestampms
    df.columns = df.columns.map(lambda x : name+'_'+x if x !='Timestampms' else x)
    
    return df
    
def merge_ato_n_others(df_ato, df_other):
    """Return a dataframe
    
    Merge ATP/COM/TDMS onto ATO
    """
    print("---Merging Dataframes---")
    # Merge other Dataframe to ATO Dataframe based on Timestampms (ordered manner)
    df_result = pd.merge_ordered(df_ato, df_other, how='outer', on='Timestampms')
    
    #Output to CSV
    #df_result.to_csv('./result_4.csv', index=False, header=True)

    # Nearest_index to indicate switch condition 0: Remain, 1:Take index after, 2: Take index before
    df_result['nearest_index'] = ""

    # Compute ATP real time and map it onto ATO while maintaining the ATP's time interval
    df_result[['Real_Timestampms', 'nearest_index']] = df_result.progress_apply(calculate_ATO_real_timestampms, df_result=df_result, axis=1)
    # Indicate which row needs to be switched
    df_result['nearest_index'] = df_result.progress_apply(indicate_switch, df_result=df_result, axis=1)
    # Switch row position 
    df_result = df_result.progress_apply(switch_row_position, df_result=df_result, axis=1)
    # Sort Real Time
    #df_result['Timestampms'] = df_result['Timestampms'].sort_values().values
    df_result = df_result.sort_values(by='Real_Timestampms',ascending=True).reset_index(drop=True)
    df_result = df_result.drop(['Timestampms', 'nearest_index'], axis=1)
    # Group resulting same real timestamp together
    df_result = df_result.groupby("Real_Timestampms").first().reset_index()
       
    #df_result.to_csv('./result_5.csv', index=False, header=True)
    df_result_ato_other = df_result.copy(deep=True)
    del df_result
    return df_result_ato_other
    
def merge_all(df_ato_atp, df_ato_com, df_ato_tdms, train_number, car_number, start_time, end_time):
    """
    
    Merge all result dataframes
    """
    print("---Merging all Dataframes (ATO, ATP, COM, TDMS)---")
    # First drop ATO columns from ATO_COM
    df_ato_com = df_ato_com.loc[:, ~df_ato_com.columns.str.startswith('ATO')]
    # duplicateRowsDF = df_result_drop_ATO_COM[df_result_drop_ATO_COM.duplicated()]
    # print(duplicateRowsDF)

    #df_result_drop_ATO_COM.info(verbose=True)
    #df_result_ATO_ATP.info(verbose=True)
    #print(df_ato_atp.shape, df_ato_com.shape)
    # Merge onto result ATO_ATP
    df_temp_result = pd.merge_ordered(df_ato_atp, df_ato_com, how='outer', on='Real_Timestampms')

    # First drop ATO columns from ATO_TDMS
    df_ato_tdms = df_ato_tdms.loc[:, ~df_ato_tdms.columns.str.startswith('ATO')]
    # Merge TDMS onto the result dataframe
    df_temp_result = pd.merge_ordered(df_temp_result, df_ato_tdms, how='outer', on='Real_Timestampms')
    
    # Rename Real_Timestampms to epoch
    df_temp_result.rename(columns={'Real_Timestampms':'epoch'}, inplace=True)
    
    start_time_new = start_time.strftime('%Y%m%d_%H%M%S')
    end_time_new = end_time.strftime('%Y%m%d_%H%M%S')
    
    filename = "OMAP_Train_"+train_number+"_Car_"+car_number+"_"+start_time_new+"_to_"+end_time_new+".csv"
    if not os.path.exists('./preprocessing_output'):
        os.makedirs('./preprocessing_output')
    
    df_temp_result.to_csv('./preprocessing_output/' + filename, index=False, header=True)
    
    return df_temp_result

def process_omap(input_folder_path, start_time, end_time):
    """
    Return merged ATO-ATP, ATO-COM, ATO-TDMS & ATO-ATP-COM-TDMS
    
    Check for folder structure, corresponding OMAP files (ATO, ATP, COM & TDMS), read logs and filter base on Date
    """
    
     # Check if EVR folder exist
    if not os.path.exists(os.path.join(input_folder_path, 'OMAP')):
        print("OMAP folder does not exist. Kindly check the folder strucutre. No OMAP processing will be performed")
        return
    else:
        print("OMAP folder exist!")
        #print(os.path.join(input_folder_path, 'OMAP'))
        pattern = "Train*"
        for entry in os.listdir(os.path.join(input_folder_path, 'OMAP')):
            #print(entry)
            if fnmatch.fnmatch(entry, pattern):
                print("Train folder exist!")
                train_folder = entry
            else:
                print("Train folder does not exist. Kindly check the folder structure. No OMAP processing will be performed")
                return
        
        train_folder_path = os.path.join(input_folder_path, 'OMAP', train_folder)
        # Retrieve Train number
        train_number = re.findall(r'%s(\d+)' %'Train ', train_folder_path)[0]
        
        # Determine time difference between start and end time
        datetime_format = '%Y/%m/%d %H:%M:%S'
        start_time = datetime.datetime.strptime(start_time, datetime_format)
        end_time = datetime.datetime.strptime(end_time, datetime_format)
        diff = end_time - start_time
        hours = diff.total_seconds() /3600

        start_time_new = start_time.strftime('%m/%d/%Y %H:%M:%S')
        end_time_new = end_time.strftime('%m/%d/%Y %H:%M:%S')
        result_list = []
        #print(os.listdir(train_folder_path))
        for car in os.listdir(train_folder_path):
            car_number = re.findall(r'%s(\d+)' %'Car ', car)[0]
            print("---Processing Train " + train_number + " Car " + car_number + "---")
            date = os.listdir(os.path.join(train_folder_path, car))[0]
            #print(date)
            flag = True
            back_date_flag = False
            # Test for correct folder structure
            try:
                datetime.datetime.strptime(date, "%y%m%d")
                #print("Correct date string format.")
            except ValueError:
                print("Incorrect date string format. Example 200116 %y%m%d. This might result in processing error.")
                flag = False    
            
            # OMAP Processing
            
            # Look for ATO, ATP, COM and TDMS folder
            if not os.path.exists(os.path.join(train_folder_path, car, date, 'OMAP_ATO')):
                print("ATO folder does not exist. Incorrect folder structure")
                flag = False
            else:
                ATO_file_list = os.listdir(os.path.join(train_folder_path, car, date, 'OMAP_ATO'))
                sorted(ATO_file_list)
                # Find dataframe/append for the given start and end time
                ATO_dataframe = pd.DataFrame()
                ATP_dataframe = pd.DataFrame()
                COM_dataframe = pd.DataFrame()
                TDMS_dataframe = pd.DataFrame()
                for index in range (0, len(ATO_file_list)):
                #for file in ATO_file_list:
                    file = ATO_file_list[index]
                    process_flag = True
                    # For each ATO file, finding corresponding ATP, COM and TDMS file
                    ATP_file_list = os.listdir(os.path.join(train_folder_path, car, date, 'OMAP_ATP'))
                    if not file.replace('ATO','ATP') in ATP_file_list:
                        print(file + " does not have a corresponding ATP file. Hence will not be processed. ")
                        process_flag = False
                    
                    COM_file_list = os.listdir(os.path.join(train_folder_path, car, date, 'OMAP_COM'))
                    if not file.replace('ATO','COM') in COM_file_list:
                        print(file + " does not have a corresponding COM file. Hence will not be processed. ")
                        process_flag = False
                        
                    TDMS_file_list = os.listdir(os.path.join(train_folder_path, car, date, 'OMAP_TDMS'))
                    if not file.replace('ATO','TDMS') in TDMS_file_list:
                        print(file + " does not have a corresponding TDMS file. Hence will not be processed. ")
                        process_flag = False               
                    
                    if process_flag:
                        # Read ATO log file and find relevant rows based on start and end time)
                        df_ATO = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_ATO', file), sep="\t")
                        df_ATO = clean_datetime(df_ATO)
                        #df_ATO['Date'] = pd.to_datetime(df_ATO['Date'], format='%m/%d/%Y %H:%M:%S')
                        # Read ATP log file and find relevant rows based on start and end time)
                        df_ATP = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_ATP', file.replace('ATO','ATP')), sep="\t")
                        df_ATP = clean_datetime(df_ATP)
                        #df_ATP['Date'] = pd.to_datetime(df_ATP['Date'], format='%m/%d/%Y %H:%M:%S')
                        # Read COM log file and find relevant rows based on start and end time)
                        df_COM = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_COM', file.replace('ATO','COM')), sep="\t")
                        df_COM = clean_datetime(df_COM)
                        #df_COM['Date'] = pd.to_datetime(df_COM['Date'], format='%m/%d/%Y %H:%M:%S')
                        # Read TDMS log file and find relevant rows based on start and end time)
                        df_TDMS = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_TDMS', file.replace('ATO','TDMS')), sep="\t")
                        df_TDMS = clean_datetime(df_TDMS)
                        #df_TDMS['Date'] = pd.to_datetime(df_TDMS['Date'], format='%m/%d/%Y %H:%M:%S')
                                            
                        if hours == 1: # Process only 1 log file
                            # Compare between cuurent index and next index (to be removed as it doesnt max sense to do it this way)
                            # For the sake of being the same as ATLAS
                            if (index+1 < len(ATO_file_list)):
                                df_ATO_next = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_ATO', ATO_file_list[index+1]), sep="\t")
                                df_ATO_next = clean_datetime(df_ATO_next)
                                #df_ATO_next['Date'] = pd.to_datetime(df_ATO_next['Date'], format='%m/%d/%Y %H:%M:%S')
                                next_length = len((df_ATO_next.loc[(df_ATO_next['Date'] == start_time_new)]))
                                current_length = len((df_ATO.loc[(df_ATO['Date'] == start_time_new)]))
                                
                                if next_length > current_length:
                                    # Read logs from next index
                                    df_ATP_next = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_ATP', ATO_file_list[index+1].replace('ATO','ATP')), sep="\t")
                                    df_ATP_next = clean_datetime(df_ATP_next)
                                    #df_ATP_next['Date'] = pd.to_datetime(df_ATP_next['Date'], format='%m/%d/%Y %H:%M:%S')
                                    df_COM_next= pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_COM', ATO_file_list[index+1].replace('ATO','COM')), sep="\t")
                                    df_COM_next = clean_datetime(df_COM_next)
                                    #df_COM_next['Date'] = pd.to_datetime(df_COM_next['Date'], format='%m/%d/%Y %H:%M:%S')
                                    df_TDMS_next = pd.read_csv(os.path.join(train_folder_path, car, date, 'OMAP_TDMS', ATO_file_list[index+1].replace('ATO','TDMS')), sep="\t")
                                    df_TDMS_next = clean_datetime(df_TDMS_next)
                                    #df_TDMS_next['Date'] = pd.to_datetime(df_TDMS_next['Date'], format='%m/%d/%Y %H:%M:%S') 
                                    
                                    ATO_dataframe = ATO_dataframe.append(df_ATO_next.loc[(df_ATO_next['Date'] >= start_time_new) & (df_ATO_next['Date'] <= end_time_new)])
                                    ATP_dataframe = ATP_dataframe.append(df_ATP_next.loc[(df_ATP_next['Date'] >= start_time_new) & (df_ATP_next['Date'] <= end_time_new)])
                                    COM_dataframe = COM_dataframe.append(df_COM_next.loc[(df_COM_next['Date'] >= start_time_new) & (df_COM_next['Date'] <= end_time_new)])
                                    TDMS_dataframe = TDMS_dataframe.append(df_TDMS_next.loc[(df_TDMS_next['Date'] >= start_time_new) & (df_TDMS_next['Date'] <= end_time_new)])
                                    break
                                else:
                                    # Read logs from current index
                                    ATO_dataframe = ATO_dataframe.append(df_ATO.loc[(df_ATO['Date'] >= start_time_new) & (df_ATO['Date'] <= end_time_new)])
                                    ATP_dataframe = ATP_dataframe.append(df_ATP.loc[(df_ATP['Date'] >= start_time_new) & (df_ATP['Date'] <= end_time_new)])
                                    COM_dataframe = COM_dataframe.append(df_COM.loc[(df_COM['Date'] >= start_time_new) & (df_COM['Date'] <= end_time_new)])
                                    TDMS_dataframe = TDMS_dataframe.append(df_TDMS.loc[(df_TDMS['Date'] >= start_time_new) & (df_TDMS['Date'] <= end_time_new)])
                                    break
                            else:
                                # Read logs from only index
                                ATO_dataframe = ATO_dataframe.append(df_ATO.loc[(df_ATO['Date'] >= start_time_new) & (df_ATO['Date'] <= end_time_new)])
                                ATP_dataframe = ATP_dataframe.append(df_ATP.loc[(df_ATP['Date'] >= start_time_new) & (df_ATP['Date'] <= end_time_new)])
                                COM_dataframe = COM_dataframe.append(df_COM.loc[(df_COM['Date'] >= start_time_new) & (df_COM['Date'] <= end_time_new)])
                                TDMS_dataframe = TDMS_dataframe.append(df_TDMS.loc[(df_TDMS['Date'] >= start_time_new) & (df_TDMS['Date'] <= end_time_new)])
                                break
                        else:
                            # Need to fix for whole number hours (probably not because dont have to copy how atlas process which files)
                            ATO_dataframe = ATO_dataframe.append(df_ATO.loc[(df_ATO['Date'] >= start_time_new) & (df_ATO['Date'] < end_time_new)])
                            ATP_dataframe = ATP_dataframe.append(df_ATP.loc[(df_ATP['Date'] >= start_time_new) & (df_ATP['Date'] < end_time_new)])
                            COM_dataframe = COM_dataframe.append(df_COM.loc[(df_COM['Date'] >= start_time_new) & (df_COM['Date'] < end_time_new)])
                            TDMS_dataframe = TDMS_dataframe.append(df_TDMS.loc[(df_TDMS['Date'] >= start_time_new) & (df_TDMS['Date'] < end_time_new)])
                            # Check if log is '00'_00_00 time and within time period
                            if re.findall(r'%s(\d+)' %'_', file)[0] == '00' and len(df_ATO.loc[(df_ATO['Date'] >= start_time_new) & (df_ATO['Date'] < end_time_new)])>0:
                                back_date_flag = True
                    else:
                        print(car + " data folder is incorrect / missing corresponding log files. Cannot proceed with processing! Kindly check whether ATO, ATP COM and TDMS have their corresponding log files.")
                        return
                
                    # Check if log is '00'_00_00 time and within time period
                    if re.findall(r'%s(\d+)' %'_', file)[0] == '00' and len(ATO_dataframe.loc[(ATO_dataframe['Date'] >= start_time_new) & (ATO_dataframe['Date'] <= end_time_new)])>0:
                        back_date_flag = True
                        print("back_date_flag")
                            
                ATO_dataframe = ATO_dataframe.reset_index(drop=True)
                ATP_dataframe = ATP_dataframe.reset_index(drop=True)
                COM_dataframe = COM_dataframe.reset_index(drop=True)
                TDMS_dataframe = TDMS_dataframe.reset_index(drop=True)
                
                if(ATO_dataframe.shape[0] == 0):
                    sys.exit("No matching time window from start to end time! Kindly check if you have input the correct start and end time!")
                
                #print(ATO_dataframe.tail(30))
                # Process ATO, ATP, COM, TDMS individually
                ATO_dataframe = process_ato(ATO_dataframe, back_date_flag)
                ATP_dataframe = process_others(ATP_dataframe, 'ATP')
                COM_dataframe = process_others(COM_dataframe, 'COM')
                TDMS_dataframe = process_others(TDMS_dataframe, 'TDMS')

                # Merge ATO-ATP, ATO-COM, ATO-TDMS
                ATO_ATP_result = merge_ato_n_others(ATO_dataframe, ATP_dataframe)
                ATO_COM_result = merge_ato_n_others(ATO_dataframe, COM_dataframe)
                ATO_TDMS_result = merge_ato_n_others(ATO_dataframe, TDMS_dataframe)

                # Merge all results dataframes
                df_result = merge_all(ATO_ATP_result, ATO_COM_result, ATO_TDMS_result, train_number, car_number, start_time, end_time)
                
                # Append results for Unit Test
                result_list.append([ATO_ATP_result, ATO_COM_result, ATO_TDMS_result, df_result])
                
    return result_list
    
def unit_test(sample_output_filename, name, df_test):
    """

    Perform Unit Test for merged dataframes
    """
    print("---Unit Test for merged ATO and " + name + " Dataframe---")
    # Unit Test for ATO to TDMS
    # Import Output File
    df_output = pd.read_csv(sample_output_filename)
    # Only retrieve respective columns
    if name == 'TDMS':
        df_output = df_output.drop(df_output[(df_output['ATO_0101__General'].isnull()) & (df_output['TDMS_002_General_Data'].isnull())].index)
        #drop column with prefix ATP and COM
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('ATP')]
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('COM')]
    elif name == 'ATP':
        df_output = df_output.drop(df_output[(df_output['ATO_0101__General'].isnull()) & (df_output['ATP_002_Loc_fault'].isnull())].index)
        #drop column with prefix TDMS and COM
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('TDMS')]
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('COM')]
    elif name == 'COM':
        df_output = df_output.drop(df_output[(df_output['ATO_0101__General'].isnull()) & (df_output['COM_002_SAFE_INPUTS'].isnull())].index)
        #drop column with prefix TDMS and COM
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('TDMS')]
        df_output = df_output.loc[:, ~df_output.columns.str.startswith('ATP')]

    df_output['epoch'] = df_output.epoch.values.astype(np.float64)
    df_output = df_output.reset_index(drop=True)
    print(df_output.shape, df_test.shape)    
    #Output to CSV
    #df_output.to_csv('./df_output.csv', index=False, header=True)

    #Output to CSV
    #df_test.to_csv('./df_test.csv', index=False, header=True)
    #df_drop_test = pd.read_csv('./df_test.csv')
    df_drop_test = df_test
    #df_test['result'] = np.where(df_test['Real_Timestampms'] == df_output['epoch'], 'True', 'False')
    #df_test.to_csv('./df_test.csv', index=False, header=True)
    #df_drop_test = df_drop_test.sort_values(by='ATO_Real_Timestampms',ascending=True).reset_index(drop=True)
    # Assert whether sample output and self processed are equal
    assert_equal = utility.nan_equal(df_drop_test['Real_Timestampms'].values, df_output['epoch'].values)
    print("Epoch Time Equality Between Sample Output and Self Processed: ", assert_equal)
    assert_equal = utility.nan_equal(df_drop_test['ATO_* General'].values, df_output['ATO_0101__General'].values)
    print("ATO General Equality Between Sample Output and Self Processed: ", assert_equal)
    df_drop_test.columns = df_output.columns
    #print(np.testing.assert_allclose(df_drop_test.values, df_output.values, rtol=1e-10, atol=0))
    #print(pd.testing.assert_frame_equal(df_drop_test, df_output, check_dtype=False))
    print("Difference betwen ATLAS Output and Self Generated: ", df_drop_test.compare(df_output, align_axis=0))
    
def unit_test_all(sample_output_filename, df_result):
    """

    Perform Unit Test for all merged results dataframes
    """
    print("---Unit Test between ATLAS Output and Self Generated---")
    # Unit Test for End Result
    # Import Output File
    df_output = pd.read_csv(sample_output_filename)
    print(df_output.shape, df_result.shape)
    assert_equal = utility.nan_equal(df_result['epoch'].values, df_output['epoch'].values)
    print("Epoch Time Equality Between Sample Output and Self Processed: ", assert_equal)
    assert_equal = utility.nan_equal(df_result['ATO_* General'].values, df_output['ATO_0101__General'].values)
    print("ATO General Equality Between Sample Output and Self Processed: ", assert_equal)
    #print(np.testing.assert_equal(df_result.values, df_output.values))
    #print(np.testing.assert_equal(df_result['epoch'].values, df_output['epoch'].values))
    #print(np.testing.assert_equal(df_result['ATO_* General'].values, df_output['ATO_0101__General'].values))
    #print(np.testing.assert_equal(df_result['ATP_Loc fault'].values, df_output['ATP_002_Loc_fault'].values))
    #print(np.testing.assert_equal(df_result['COM_SAFE INPUTS'].values, df_output['COM_002_SAFE_INPUTS'].values))
    #print(np.testing.assert_equal(df_result['TDMS_General Data'].values, df_output['TDMS_002_General_Data'].values))
    #print(np.testing.assert_equal(df_result['TDMS_Sec'].values, df_output['TDMS_016_Sec'].values))

    # Energry Delta has some error (show in powerpoint slides).
    #print(np.testing.assert_equal(df_result['ATO_Energy delta'].values, df_output['ATO_1220_Energy_delta'].values))

    #print(np.testing.assert_allclose(df_result.values, df_output.values, rtol=1e-10, atol=0))
    #print(np.testing.assert_equal(df_result.values, df_output.values))
    df_result_2 = df_result.copy(deep=True)
    df_result_2.columns = df_output.columns
    #print(pd.testing.assert_frame_equal(df_result_2, df_output, check_dtype=False, check_column_type=False))
    print("Difference betwen ATLAS Output and Self Generated: ", df_result_2.compare(df_output, align_axis=0))