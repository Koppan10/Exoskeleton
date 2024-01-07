#CODE DESCRIPTION:
#
#Purpose: Slopes
#


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
from sklearn.preprocessing import StandardScaler#importing sklearn
from sklearn.model_selection import train_test_split

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



# PLOT IN TIME DOMAIN

# Plot signals from ACCdata_3min
# plt.figure(figsize=(12, 6))
# plt.plot(ACCdata_file6['time'] / 1000, ACCdata_file6['emgch1'])  # Time in ms / 1000 to get it in seconds
# plt.xlabel('Time (s)')
# plt.ylabel('EMG CH1')
# plt.title('EMG Signal - file6')
# plt.show()

# Plot signals from ACCdata_2min
# plt.figure(figsize=(12, 6))
# plt.plot(ACCdata_file7['time'] / 1000, ACCdata_file7['emgch1'])  # Time in ms / 1000 to get it in seconds
# plt.xlabel('Time (s)')
# plt.ylabel('EMG CH1')
# plt.title('EMG Signal - file7')
# plt.show()


#SECTION 2: Data filtering

################ Execute remove_mean function ##########
emg_correctmean_no_weight = emgf.remove_mean(ACCdata_no_weight['emgch1'], ACCdata_no_weight['time'] / 1000) 
emg_correctmean_mvc = emgf.remove_mean(ACCdata_mvc['emgch1'], ACCdata_mvc['time'] / 1000) 
emg_correctmean_weight = emgf.remove_mean(ACCdata_weight['emgch1'], ACCdata_weight['time'] / 1000)

################################   Execute remove_mean function ####################################
emg_correctmean_weight_1kg = emgf.remove_mean(ACCdata_weight_1kg['emgch1'], ACCdata_weight_1kg['time'] / 1000)
emg_correctmean_weight_3kg = emgf.remove_mean(ACCdata_weight_3kg['emgch1'], ACCdata_weight_3kg['time'] / 1000)
emg_correctmean_weight_4kg = emgf.remove_mean(ACCdata_weight_5kg['emgch1'], ACCdata_weight_5kg['time'] / 1000)
emg_correctmean_weight_7kg = emgf.remove_mean(ACCdata_weight_7kg['emgch1'], ACCdata_weight_7kg['time'] / 1000)
emg_correctmean_weight_8kg = emgf.remove_mean(ACCdata_weight_8kg['emgch1'], ACCdata_weight_8kg['time'] / 1000)
emg_correctmean_weight_10kg = emgf.remove_mean(ACCdata_weight_10kg['emgch1'], ACCdata_weight_10kg['time'] / 1000)



################ Execute alltogether function ##########
########### This function has the flowing:emg_filtered, emg_envelope, emg_rectified
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



# PLOT IN TIME DOMAIN

#Plot signals from ACCdata_3min
plt.figure(figsize=(12, 6))
plt.plot(emg_envelope_weight_4kg,color='blue')  # Time in ms / 1000 to get it in seconds
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EMG Envelope Signal')
plt.grid(True)


# Assuming emg_envelope_signal contains the EMG envelope signal (NumPy array)

# Calculate time differences between samples (assuming uniform time intervals)
time_diff = 1  # Change this value if time intervals are different
time = np.arange(len(emg_envelope_no_weight)) * time_diff

# Calculate slopes (numerical differentiation)
slopes = np.gradient(emg_envelope_no_weight, time)

# Now 'slopes' contains the slope at each time point in the EMG envelope signal


# Plotting the slopes in the time domain
plt.figure(figsize=(10, 6))
plt.plot(time, slopes, color='green')
plt.title('Slope Changes in EMG Envelope Signal')
plt.xlabel('Time')
plt.ylabel('Slope')
plt.grid(True)


import numpy as np
import matplotlib.pyplot as plt

# Assuming emg_envelope_signal contains the EMG envelope signal (NumPy array)

# Calculate a lower threshold for detecting contractions (e.g., 70th percentile)
threshold = np.percentile(emg_envelope_mvc, 50)  # Adjust percentile as needed

# Identify where the signal crosses the lower threshold
crossings = np.where(emg_envelope_mvc > threshold)[0]

# Find the start and end points of contraction segments
start_points = []
end_points = []

if len(crossings) > 0:
    start_points.append(crossings[0])

    for i in range(1, len(crossings)):
        if crossings[i] != crossings[i - 1] + 1:
            end_points.append(crossings[i - 1])
            start_points.append(crossings[i])

    end_points.append(crossings[-1])


#HERE    

# Visualize the EMG envelope signal and highlight contraction segments
plt.figure(figsize=(10, 6))
plt.plot(emg_envelope_mvc, color='blue', label='EMG Envelope Signal')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

# Highlight detected contraction segments
for start, end in zip(start_points, end_points):
    plt.axvspan(start, end, color='green', alpha=0.3, label='Contraction')

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('EMG Envelope Signal with Detected Contractions (Lower Threshold)')
plt.legend()
plt.grid(True)
plt.show()














