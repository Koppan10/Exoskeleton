
#CODE DESCRIPTION:
#
#Purpose: To find a relationship between amplitude on EMG signals in mV and different loads/weights on the arm. 
#This expirement is based only on bicep muscle contraction
#The code plots emg unfiltered in time and frequency and then some data is plotted in filtered version.

#The code is divided into 7 Sections. Each section is presesnted when it starts throughout the code and the aim od the section is stated.

import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model

import EMG_functions as emgf
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
import pywt

#SECTION 1: Data acquisition 


# #Sampling frequency
sf = 1024

# # 1 DATA COLLECTION

 #Training and testind data

file2 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session2\emg_no_weight.txt'
column_names = ['time', 'emgch1']
ACCdata_no_weight = pd.read_csv(file2, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')

file = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session1\full_contraction_emg.txt'
column_names = ['time', 'emgch1']
ACCdata_mvc = pd.read_csv(file, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')

file3 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session3\emg_with_weight.txt'
column_names = ['time', 'emgch1']
ACCdata_weight = pd.read_csv(file3, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python') 

file4 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session4\2023-10-16_11Read_muscle_3min_weight\Assessment1_Session1_Shimmer_9D70_Calibrated_PC.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_7kg = pd.read_csv(file4, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file5 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\2023-11-08_Jalal_emg\Session2_7.5kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_8kg = pd.read_csv(file5, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file6 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\2023-11-08_Jalal_emg\Session3_5kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_5kg = pd.read_csv(file6, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file7 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\2023-11-08_Jalal_emg\Session4_2.5kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_3kg = pd.read_csv(file7, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file8 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\2023-11-08_Jalal_emg\Session5_1.25kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_1kg = pd.read_csv(file8, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file9 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\2023-11-08_Jalal_emg\Session1_10kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_10kg = pd.read_csv(file9, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  



#PLOT IN TIME DOMAIN

#Plot signals from ACCdata_3min
plt.figure(figsize=(12, 6))
plt.plot(ACCdata_weight_10kg['time'] / 1000, ACCdata_weight_10kg['emgch1'])  # Time in ms / 1000 to get it in seconds
plt.xlabel('Time (s)')
plt.ylabel('EMG CH1')
plt.title('EMG Signal')
plt.show()

#Plot signals from ACCdata_2min
plt.figure(figsize=(12, 6))
plt.plot(ACCdata_weight_10kg['time'] / 1000, ACCdata_weight_10kg['emgch1'], color='red')  # Time in ms / 1000 to get it in seconds
plt.xlabel('Time (s)')
plt.ylabel('EMG CH1')
plt.title('EMG Signal')
plt.show()


####Frequency domain features#####

# Calculate FFT of the EMG signal
frequencies, magnitude = emgf.calculate_fft(ACCdata_weight_10kg['emgch1'], sf)


# Visualize the frequency domain
view_frequency_domain=emgf.visualize_frequency_domain(frequencies, magnitude, xlim=(0, 550), ylim=(0, 1500))

from scipy import signal

# Extract EMG data from the DataFrame
emg_data = ACCdata_weight_10kg['emgch1'].values

# Calculate Power Spectral Density (PSD)
frequencies, psd = signal.welch(emg_data, fs=1000)  # Assuming a sampling frequency of 1000 Hz, modify 'fs' if it's different

# Plot the PSD
plt.figure(figsize=(8, 4))
plt.semilogy(frequencies, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density (PSD) of EMG Data')
plt.grid(True)
plt.show()





# Extract EMG data from the DataFrame
emg_data = ACCdata_weight_3kg['emgch1'].values

# Perform Continuous Wavelet Transform (CWT)
wavelet = 'cmor'  # Choosing a specific wavelet, 'cmor' in this case
scales = np.arange(1, 128)  # Range of scales for analysis
coefficients, frequencies = pywt.cwt(emg_data, scales, wavelet)

# Plot the Continuous Wavelet Transform
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), extent=[0, len(emg_data), 1, 128], aspect='auto', cmap='jet')
plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.title('Continuous Wavelet Transform (CWT) of EMG Data')
plt.show()



# #SECTION 2: Data filtering

# ################ Execute remove_mean function ##########
emg_correctmean_no_weight = emgf.remove_mean(ACCdata_no_weight['emgch1'], ACCdata_no_weight['time'] / 1000) 
emg_correctmean_mvc = emgf.remove_mean(ACCdata_mvc['emgch1'], ACCdata_mvc['time'] / 1000) 
emg_correctmean_weight = emgf.remove_mean(ACCdata_weight['emgch1'], ACCdata_weight['time'] / 1000)

# ################################   Execute remove_mean function ####################################
emg_correctmean_weight_1kg = emgf.remove_mean(ACCdata_weight_1kg['emgch1'], ACCdata_weight_1kg['time'] / 1000)
emg_correctmean_weight_3kg = emgf.remove_mean(ACCdata_weight_3kg['emgch1'], ACCdata_weight_3kg['time'] / 1000)
emg_correctmean_weight_4kg = emgf.remove_mean(ACCdata_weight_5kg['emgch1'], ACCdata_weight_5kg['time'] / 1000)
emg_correctmean_weight_7kg = emgf.remove_mean(ACCdata_weight_7kg['emgch1'], ACCdata_weight_7kg['time'] / 1000)
emg_correctmean_weight_8kg = emgf.remove_mean(ACCdata_weight_8kg['emgch1'], ACCdata_weight_8kg['time'] / 1000)
emg_correctmean_weight_10kg = emgf.remove_mean(ACCdata_weight_10kg['emgch1'], ACCdata_weight_10kg['time'] / 1000)

################ Execute alltogether function ##########
############ This function has the flowing:emg_filtered, emg_envelope, emg_rectified
emg_filtered_mvc, emg_envelope_mvc, emg_rectified_mvc= emgf.alltogether (ACCdata_mvc['time'] / 1000, emg_correctmean_mvc )# sfreq has the sampling frequency 
emg_filtered_no_weight, emg_envelope_no_weight, emg_rectified_no_weight= emgf.alltogether (ACCdata_no_weight['time'] / 1000,  emg_correctmean_no_weight)
emg_filtered_weight, emg_envelope_weight, emg_rectified_weight= emgf.alltogether (ACCdata_weight['time'] / 1000, emg_correctmean_weight)




################################   Execute alltogether function ####################################

emg_filtered_weight_1kg, emg_envelope_weight_1kg, emg_rectified_weight_1kg= emgf.alltogether (ACCdata_weight_1kg['time'] / 1000, emg_correctmean_weight_1kg)
emg_filtered_weight_3kg, emg_envelope_weight_3kg, emg_rectified_weight_3kg= emgf.alltogether (ACCdata_weight_3kg['time'] / 1000, emg_correctmean_weight_3kg)
emg_filtered_weight_4kg, emg_envelope_weight_4kg, emg_rectified_weight_4kg= emgf.alltogether (ACCdata_weight_5kg['time'] / 1000, emg_correctmean_weight_4kg)
emg_filtered_weight_7kg, emg_envelope_weight_7kg, emg_rectified_weight_7kg= emgf.alltogether (ACCdata_weight_7kg['time'] / 1000, emg_correctmean_weight_7kg)
emg_filtered_weight_8kg, emg_envelope_weight_8kg, emg_rectified_weight_8kg= emgf.alltogether (ACCdata_weight_8kg['time'] / 1000, emg_correctmean_weight_8kg)
emg_filtered_weight_10kg, emg_envelope_weight_10kg, emg_rectified_weight_10kg= emgf.alltogether (ACCdata_weight_10kg['time'] / 1000, emg_correctmean_weight_10kg)


#Plot signals from ACCdata_3min
plt.figure(figsize=(12, 6))
plt.plot(ACCdata_mvc['time'] / 1000, emg_filtered_mvc, color="orange")  # Time in ms / 1000 to get it in seconds
plt.xlabel('Time (s)')
plt.ylabel('EMG CH1')
plt.title('EMG Signal')
plt.show()

#SECTION 3: Feature extraction

################## Features extraction USING emg_filtered and emg_rectified######################

# Threshold value for labeling
#threshold = np.mean(emg_rectified_weight)  # Define your threshold
threshold = np.percentile(emg_envelope_weight, 44)  # Adjust percentile as needed
print(threshold)
################## Features extraction  USING emg_filtered and emg_rectified  ######################
mav_values_mvc, rms_values_mvc, wl_values_mvc, zc_values_mvc, window_labels_mvc= emgf.emg_td_features(emg_rectified_mvc, 50, 10,threshold)
mav_values_no_weight, rms_values_no_weight, wl_values_no_weight, zc_values_no_weight, window_labels_no_weight= emgf.emg_td_features(emg_rectified_no_weight, 50, 10,threshold)
mav_values_weight, rms_values_weight, wl_values_weight, zc_values_weight, window_labels_weight= emgf.emg_td_features(emg_rectified_weight, 50, 10,threshold)

mav_values_weight_1kg, rms_values_weight_1kg, wl_values_weight_1kg, zc_values_weight_1kg, window_labels_weight_1kg= emgf.emg_td_features(emg_rectified_weight_1kg, 50, 10,threshold)
mav_values_weight_3kg, rms_values_weight_3kg, wl_values_weight_3kg, zc_values_weight_3kg, window_labels_weight_3kg= emgf.emg_td_features (emg_rectified_weight_3kg, 50, 10,threshold)
mav_values_weight_4kg, rms_values_weight_4kg, wl_values_weight_4kg, zc_values_weight_4kg, window_labels_weight_4kg= emgf.emg_td_features (emg_rectified_weight_4kg, 50, 10,threshold)
mav_values_weight_7kg, rms_values_weight_7kg, wl_values_weight_7kg, zc_values_weight_7kg, window_labels_weight_7kg= emgf.emg_td_features(emg_rectified_weight_7kg, 50, 10,threshold)
mav_values_weight_8kg, rms_values_weight_8kg, wl_values_weight_8kg, zc_values_weight_8kg, window_labels_weight_8kg= emgf.emg_td_features (emg_rectified_weight_8kg, 50, 10,threshold)
mav_values_weight_10kg, rms_values_weight_10kg, wl_values_weight_10kg, zc_values_weight_10kg, window_labels_weight_10kg= emgf.emg_td_features (emg_rectified_weight_10kg, 50, 10,threshold)


####### Plot all emg_envelopes of different signals #####

names1 = ['Envelope mvc', 'Envelope no weight', 'Envelope 3kg']  # Provide names for each envelope
time_axis = range(0, len(emg_envelope_no_weight))
emg_envelopes = [emg_envelope_mvc, emg_envelope_no_weight, emg_envelope_weight]
view_envelopes1=emgf.plot_emg_envelopes_common_time(time_axis, emg_envelopes, names1)



names2 = ['Envelope 1kg', 'Envelope 3kg', 'Envelope 7kg']  # Provide names for each envelope
time_axis = range(0, len(emg_envelope_weight_1kg))
emg_envelopes_2 = [emg_envelope_weight_1kg, emg_envelope_weight_3kg,emg_envelope_weight_7kg]
view_envelopes2=emgf.plot_emg_envelopes_common_time(time_axis, emg_envelopes_2, names2)




####### I made this plot to know the time for each envelope############

# print("Length of time_axis:", len(time_axis))
# for i, envelope in enumerate(emg_envelopes):
#     print(f"Length of emg_envelope {i + 1}:", len(envelope))


#HERE

#EMG rms amplitudes for different weights
weights = [1.25, 2.5, 5, 7.5, 10]  
emg_amplitudes = [rms_values_weight_1kg, rms_values_weight_3kg, rms_values_weight_4kg, rms_values_weight_8kg, rms_values_weight_10kg]  # EMG amplitude data for each weight

plt.figure(figsize=(8, 6))
for i, weight in enumerate(weights):
    plt.scatter([weight] * len(emg_amplitudes[i]), emg_amplitudes[i], label=f'Weight {weight}')


#Scatter plot

plt.xlabel('Weights')
plt.ylabel('EMG Amplitude')
plt.title('EMG Amplitude vs. Different Weights')
plt.legend()
plt.grid(True)
plt.show()

# #Box plot (Box-and-Whisker plot)

plt.figure(figsize=(8, 6))
plt.boxplot(emg_amplitudes, labels=weights)
plt.xlabel('Weights')
plt.ylabel('EMG Amplitude')
plt.title('EMG Amplitude Distribution Across Different Weights')
plt.grid(True)
plt.show()






