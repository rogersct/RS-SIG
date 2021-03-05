import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np
import pandas as pd
from statistics import mean 
import scipy.signal
#print(scipy.__version__)
from scipy.signal import find_peaks, peak_prominences
import collections 
import seaborn as sns
import datetime

sns.set(rc={'figure.figsize':(20, 10)})

def top_x_prominences(speed, top_num_candidate, name):
    """Return a peaks location, height of promiences, and top x promineces indices
    
    Find peaks / prominences of raw speed profile
    """
    if name == "OMAP":
        distance = 1000
    elif name == "EVR":
        distance = 200
    # Find peak
    peaks, properties = find_peaks(speed, distance=distance)
    
    # Find top x prominences
    prominences = peak_prominences(speed, peaks)[0]
    # Sort Descending and retrieve top x
    top_x_prominences_index =  prominences.argsort()[-top_num_candidate:][::-1]
    
    contour_heights = speed[peaks] - prominences
    
    #print(peaks, contour_heights, top_x_prominences_index)
    return peaks, contour_heights, top_x_prominences_index

def estimate_time_shift(omap_df, evr_df):
    """Return estimated time shift
    
    Based on the top x promineces detected, find matching peaks and vote based on epoch difference and estimate the mean time shift
    """
    print("---Estimating time shift between OMAP and EVR---")
    omap_df_speed = omap_df['ATO_Train speed'].interpolate().values
    evr_df_speed = evr_df['EOD_V1_OA_DMASPE'].interpolate().values
    # omap_df_speed = omap_df['ATO_0803_Train_speed'].interpolate().values
    # evr_df_speed = evr_df['EOD_179_V1_OA_DMASPE'].interpolate().values
    top_num_candidate = 10
    # Find OMAP Speed Peak Prominences
    omap_peaks, omap_contour_heights, omap_top_x_prominences_index = top_x_prominences(omap_df_speed, top_num_candidate, "OMAP")
    # Find EVR Speed Peak Prominences
    evr_peaks, evr_contour_heights, evr_top_x_prominences_index = top_x_prominences(evr_df_speed, top_num_candidate, "EVR")
    
    # Find differences between peak prominences
    differences_list = []
    for index in omap_top_x_prominences_index:
        omap_timestamp = omap_df.iloc[omap_peaks[index]]['epoch']
        for y_index in evr_top_x_prominences_index:
            evr_timestamp = evr_df.iloc[evr_peaks[y_index]]['epoch']
            diff = omap_timestamp-evr_timestamp
            if abs(diff) <= 25000:
                differences_list.append(diff)
    
    if len(differences_list) == 0:
        print("OMAP and EVR Speed Profile doesn't overlap. They're not of the same time period! Hence, cannot estimate the time shift.")
        mean_time_shift = 0
    else:
        differences_list_div = np.divide(differences_list, 1000).astype(int)
       
        # occurence_count = collections.Counter(differences_list_div) 
        # most_common = occurence_count.most_common(1)[0][0]
        index_list = []
        if 0 in differences_list_div:
            #Find index
            index_list = [i for i, j in enumerate(differences_list_div) if j == 0]
        else:
            index_list = [i for i, j in enumerate(differences_list_div) if j != 0]
        
        #print(differences_list_div)
        #print(index_list)
        indexed_differences_list = map(lambda i: differences_list[i], index_list)
        mean_time_shift = int(mean(indexed_differences_list))
        #print(differences_list, mean_time_shift)
    
    fig, axs = plt.subplots(nrows=3,sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.7)
    ax = axs[0]
    ax.plot(omap_df['epoch'].values, omap_df_speed, 'b')
    ax.plot(omap_df['epoch'].values[omap_peaks], omap_df_speed[omap_peaks], 'gx')
    ax.plot(omap_df['epoch'].values[omap_peaks[omap_top_x_prominences_index]], omap_df_speed[omap_peaks[omap_top_x_prominences_index]], 'kx')
    ax.vlines(x=omap_df['epoch'].values[omap_peaks[omap_top_x_prominences_index]], ymin=omap_contour_heights[omap_top_x_prominences_index], ymax=omap_df_speed[omap_peaks[omap_top_x_prominences_index]], color='k')
    ax.set_title ("OMAP Peaks and Top {} Prominences" .format(top_num_candidate))
    top_str = "Top {} Prominences" .format(top_num_candidate)
    ax.legend(["Raw Speed", "Peaks", top_str], loc="upper left")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw Speed')
    ax.grid(True)
    plt.show()
    
    ax = axs[1]
    ax.plot(evr_df['epoch'].values, evr_df_speed, 'r')
    ax.plot(evr_df['epoch'].values[evr_peaks], evr_df_speed[evr_peaks], 'gx')
    ax.plot(evr_df['epoch'].values[evr_peaks[evr_top_x_prominences_index]], evr_df_speed[evr_peaks[evr_top_x_prominences_index]], 'kx')
    ax.vlines(x=evr_df['epoch'].values[evr_peaks[evr_top_x_prominences_index]], ymin=evr_contour_heights[evr_top_x_prominences_index], ymax=evr_df_speed[evr_peaks[evr_top_x_prominences_index]], color='k')
    ax.set_title ("EVR Peaks and Top {} Prominences" .format(top_num_candidate))
    top_str = "Top {} Prominences" .format(top_num_candidate)
    ax.legend(["Raw Speed", "Peaks", top_str], loc="upper left")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw Speed')
    ax.grid(True)
    
    ax = axs[2]
    ax.plot(omap_df['epoch'].values, omap_df_speed, 'b')
    ax.plot(evr_df['epoch'].values, evr_df_speed, 'r')
    ax.plot(evr_df['epoch'].values + mean_time_shift, evr_df_speed, 'g')
    ax.set_title ("OMAP & EVR Time Syncrhonized, Time Shift = {:.3f}ms".format(mean_time_shift))
    ax.legend(["ATO_Train_speed", "EOD_V1_OA_DMASPE", "EOD_V1_OA_DMASPE_Corrected"], loc="upper left")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw Speed')
    ax.grid(True)
    
    # Raw Speed Profile
    raw_mean_acc = compute_speed_accuracy (omap_df, evr_df)
    
    # Corrected Speed Profile
    #evr_df['epoch'] = evr_df['epoch'].apply(lambda x : x+mean_time_shift)
    corrected_mean_acc = compute_speed_accuracy (omap_df, evr_df)
    
    #print(raw_mean_acc, corrected_mean_acc)
    
    
    return mean_time_shift
    

def apply_time_shift(df, time_shift, evr_filepath):
    """
    
    Based on the top x promineces detected, find matching peaks and vote based on epoch difference and estimate the mean time shift
    """
    print("---Applying time shift on EVR---")
    # Apply Time shift to EVR Epoch
    df['epoch'] = df['epoch'].apply(lambda x : x+time_shift)
    filename = evr_filepath.split('.csv')[0] + "_Corrected.csv"
    df.to_csv(filename, index=False, header=True)
    
    return df
    
def compute_speed_accuracy (omap_df, evr_df):
    """ Return Speed Accuracy
    
    Based on the top x promineces detected, find matching peaks and vote based on epoch difference and estimate the mean time shift
    """
    #Set Index of dataframe to epoch
    omap_df = omap_df.set_index('epoch')
    omap_df = omap_df['ATO_Train speed'].to_frame()
    #omap_df = omap_df['ATO_0803_Train_speed'].to_frame()
    #evr_df['epoch'] = evr_df['epoch'].apply(lambda x : x+mean_time_shift)
    evr_df = evr_df.set_index('epoch')
    evr_df = evr_df['EOD_V1_OA_DMASPE'].to_frame()
    #evr_df = evr_df['EOD_179_V1_OA_DMASPE'].to_frame()
 
    combined = omap_df.join(evr_df, how='outer')
    combined.index = pd.to_datetime(combined.index,unit='ms')

    combined = combined.apply(pd.Series.interpolate, args=('time',))
    #combined = combined.fillna(method='backfill')
    
    combined['acc_test'] = abs(combined['ATO_Train speed'] - combined['EOD_V1_OA_DMASPE'])
    #combined['acc_test'] = abs(combined['ATO_0803_Train_speed'] - combined['EOD_179_V1_OA_DMASPE'])
    #print(combined['acc_test'].mean())
    return(combined['acc_test'].mean())

def display_overrun (omap_df, evr_df):
    """For Demonstration Purposes
    
    """
    omap_df_epoch = omap_df['epoch'].values
    omap_df_epoch_datetime = [datetime.datetime.utcfromtimestamp(epoch / 1000.).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for epoch in omap_df_epoch]
    #datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
    #Gives the number of days (fraction part represents hours, minutes, seconds) since 0001-01-01 00:00:00 UTC, plus one
    omap_df_epoch_datetime_num = matplotlib.dates.date2num(omap_df_epoch_datetime)
    #omap_df_epoch_datetime = np.array(omap_df_epoch[0], dtype='i8').view('datetime64[ms]').tolist()
    print(omap_df_epoch_datetime_num[0],omap_df_epoch_datetime_num[20])
    print(omap_df_epoch_datetime[0],omap_df_epoch_datetime[20])
    #print(datetime.datetime.utcfromtimestamp(omap_df_epoch[0]).strftime('%Y-%m-%d %H:%M:%S'))
    # Plot OMAP
    fig, axs = plt.subplots(nrows=2,sharex=True) #sharey=True
    ax = axs[0]
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_0504_Area_SSP_dist'].interpolate().values, '-')
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_0803_Train_speed'].interpolate().values, '-')
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_1804_MBC'].interpolate().values, '-')
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATP_150_ZVI'].interpolate().values, '-')
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATP_145_ZVRD'].interpolate().values, '-')
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_1222_Act_RP_act_C_speed'].interpolate().values, '-')
    ax.set_title ("OMAP")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend(["ATO_0504_Area_SSP_dist", "ATO_0803_Train_speed", "ATO_1804_MBC", "ATP_150_ZVI", "ATP_145_ZVRD", "ATO_1222_Act_RP_act_C_speed"], loc="upper left")
    formatter = dates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(formatter)    
    #plt.gca().xaxis.set_major_locator(dates.HourLocator(byhour=[0,1]))
    #ax.xaxis.set_major_formatter(formatter)
    #ax.fmt_xdata = DateFormatter('% Y-% m-% d % H:% M:% S') 

    # Plot EVR
    evr_df_epoch = evr_df['epoch'].values
    evr_df_epoch_datetime = [datetime.datetime.utcfromtimestamp(epoch / 1000.).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for epoch in evr_df_epoch]
    #datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
    evr_df_epoch_datetime_num = matplotlib.dates.date2num(evr_df_epoch_datetime)
    ax = axs[1]
    ax.plot_date(evr_df_epoch_datetime_num, evr_df['EOD_179_V1_OA_DMASPE'].interpolate().values, '-')
    #EC2_205_SI1_DM_ATC_RSC
    #EC2_206_SI6_DM_ATC_RSC
    
    ax.set_title ("EVR")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    formatter = dates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(formatter) 
    fig.autofmt_xdate()
    plt.show()
  
def display_overrun_demo (omap_df, evr_df):
    """For Demonstration Purposes
    
    """
    # Convert OMAP Unix Timestamp to DateTime string
    omap_df_epoch = omap_df['epoch'].values
    omap_df_epoch_datetime = [datetime.datetime.utcfromtimestamp(epoch / 1000.).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for epoch in omap_df_epoch]
    # Give the number of days (fraction part represents hours, minutes, seconds) since 0001-01-01 00:00:00 UTC, plus one
    omap_df_epoch_datetime_num = matplotlib.dates.date2num(omap_df_epoch_datetime)
    
    # Convert EVR Unix Timestamp to DateTime string
    evr_df_epoch = evr_df['epoch'].values
    evr_df_epoch_datetime = [datetime.datetime.utcfromtimestamp(epoch / 1000.).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for epoch in evr_df_epoch]
    evr_df_epoch_datetime_num = matplotlib.dates.date2num(evr_df_epoch_datetime)
    
    #plt.style.use('dark_background')
    #plt.style.use('seaborn')
    sns.set_style("darkgrid")
    fig = plt.figure()
    gs = fig.add_gridspec(6, hspace=0)
    axs = gs.subplots(sharex=True)
    # Plot ATO_0803_Train_speed
    #fig, axs = plt.subplots(nrows=6,sharex=True) #sharey=True
    ax = axs[0]
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_Train speed'].interpolate().values, '-', color="red", label='ATO_Train speed')
    #ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_0803_Train_speed'].interpolate().values, '-', color="red", label='ATO_0803_Train_speed')
    ax.set_xlabel('Epoch')
    #ax.set_ylabel('ATO_0803_Train_speed')
    
    # Plot EOD_179_V1_OA_DMASPE
    ax = axs[1]
    ax.plot_date(evr_df_epoch_datetime_num, evr_df['EOD_V1_OA_DMASPE'].interpolate().values, '-', color="gold", label='EOD_V1_OA_DMASPE')
    #ax.plot_date(evr_df_epoch_datetime_num, evr_df['EOD_179_V1_OA_DMASPE'].interpolate().values, '-', color="yellow", label='EOD_179_V1_OA_DMASPE')
    ax.set_xlabel('Epoch')
    #ax.set_ylabel('EOD_179_V1_OA_DMASPE')
    
    # Plot ATO_1804_MBC
    ax = axs[2]
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_MBC'].interpolate().values, '-', color="green", label='ATO_MBC')
    #ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATO_1804_MBC'].interpolate().values, '-', color="green", label='ATO_1804_MBC')
    ax.set_xlabel('Epoch')
    #ax.set_ylabel('ATO_1804_MBC')
    
    # Plot EC2_207_V2_TB_TRBREFF
    ax = axs[3]
    ax.plot_date(evr_df_epoch_datetime_num, evr_df['EC2_V2_TB_TRBREFF'].interpolate().values, '-', color="purple", label='EC2_V2_TB_TRBREFF')
    #ax.plot_date(evr_df_epoch_datetime_num, evr_df['EC2_207_V2_TB_TRBREFF'].interpolate().values, '-', color="purple", label='EC2_207_V2_TB_TRBREFF')
    ax.set_xlabel('Epoch')
    #ax.set_ylabel('EC2_207_V2_TB_TRBREFF')
    
    # Plot ATP_150_ZVI
    ax = axs[4]
    ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATP_ZVI'].interpolate().values, '-', color="blue", label='ATP_ZVI')
    #ax.plot_date(omap_df_epoch_datetime_num, omap_df['ATP_150_ZVI'].interpolate().values, '-', color="blue", label='ATP_150_ZVI')
    ax.set_xlabel('Epoch')
    #ax.set_ylabel('ATP_150_ZVI')
    
    # Plot EC2_035_V1_DM_ATC1_ZVI
    ax = axs[5]
    ax.plot_date(evr_df_epoch_datetime_num, evr_df['EC2_V1_DM_ATC1_ZVI'].interpolate().values, '-', color="orange", label='EC2_V1_DM_ATC1_ZVI')
    #ax.plot_date(evr_df_epoch_datetime_num, evr_df['EC2_035_V1_DM_ATC1_ZVI'].interpolate().values, '-', color="orange", label='EC2_035_V1_DM_ATC1_ZVI')
    ax.set_xlabel('Epoch')
    #ax.set_ylabel('EC2_035_V1_DM_ATC1_ZVI')
  
    formatter = dates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(formatter)    
    fig.autofmt_xdate()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper left')
    plt.show()