import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean 
import scipy.signal
#print(scipy.__version__)
from scipy.signal import find_peaks, peak_prominences
import collections 
import seaborn as sns
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
    # omap_df_speed = omap_df['ATO_Train speed'].interpolate().values
    # evr_df_speed = evr_df['EOD_V1_OA_DMASPE'].interpolate().values
    omap_df_speed = omap_df['ATO_0803_Train_speed'].interpolate().values
    evr_df_speed = evr_df['EOD_179_V1_OA_DMASPE'].interpolate().values
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
        print(differences_list, mean_time_shift)
    
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
    evr_df['epoch'] = evr_df['epoch'].apply(lambda x : x+mean_time_shift)
    corrected_mean_acc = compute_speed_accuracy (omap_df, evr_df)
    
    print(raw_mean_acc, corrected_mean_acc)
    
    
    return mean_time_shift
    

def apply_time_shift(df, time_shift, evr_filepath):
    """
    
    Based on the top x promineces detected, find matching peaks and vote based on epoch difference and estimate the mean time shift
    """
    # Apply Time shift to EVR Epoch
    df['epoch'] = df['epoch'].apply(lambda x : x+time_shift)
    filename = evr_filepath.split('.csv')[0] + "_Corrected.csv"
    df.to_csv(filename, index=False, header=True)
    
def compute_speed_accuracy (omap_df, evr_df):
    """ Return Speed Accuracy
    
    Based on the top x promineces detected, find matching peaks and vote based on epoch difference and estimate the mean time shift
    """
    #Set Index of dataframe to epoch
    omap_df = omap_df.set_index('epoch')
    omap_df = omap_df['ATO_0803_Train_speed'].to_frame()
    #evr_df['epoch'] = evr_df['epoch'].apply(lambda x : x+mean_time_shift)
    evr_df = evr_df.set_index('epoch')
    evr_df = evr_df['EOD_179_V1_OA_DMASPE'].to_frame()
 
    combined = omap_df.join(evr_df, how='outer')
    combined.index = pd.to_datetime(combined.index,unit='ms')

    combined = combined.apply(pd.Series.interpolate, args=('time',))
    #combined = combined.fillna(method='backfill')
    
    combined['acc_test'] = abs(combined['ATO_0803_Train_speed'] - combined['EOD_179_V1_OA_DMASPE'])
    print(combined['acc_test'].mean())
    return(combined['acc_test'].mean())
