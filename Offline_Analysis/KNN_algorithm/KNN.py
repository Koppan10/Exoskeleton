#CODE DESCRIPTION:
#
#Purpose: K-Nearest Neighbor model to detect muscle contraction 
#
#The code is divided into 5 Sections. Each section is presesnted when it starts throughout the code and the aim od the section is stated.

import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import EMG_functions as emgf
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import sys
import struct
import serial 
import matplotlib.pyplot as plt
import numpy as np

from EMG_functions import remove_mean, alltogether ,emg_td_features


#SECTION 1: Data acquisition 

# #Sampling frequency
sf = 1024

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

#SECTION 3: Feature extraction


################## Features extraction USING emg_filtered and emg_rectified######################

# Threshold value for labeling

#threshold = np.mean(emg_envelope_no_weight)
threshold = np.percentile(emg_envelope_no_weight, 49.5)


# Calculate a lower threshold for detecting contractions (e.g., 70th percentile)
#Specific threshold for each signal 
# threshold_mvc = np.percentile(emg_envelope_mvc, 50)  # Adjust percentile as needed
# threshold_no_weight = np.percentile(emg_envelope_no_weight, 49.5)
# threshold_weight = np.percentile(emg_envelope_weight, 46)

# threshold_weight_1kg = np.percentile(emg_envelope_weight_1kg, 47.5)
# threshold_weight_3kg = np.percentile(emg_envelope_weight_3kg, 56.5)
# threshold_weight_4kg = np.percentile(emg_envelope_weight_4kg, 56.5)
# threshold_weight_7kg = np.percentile(emg_envelope_weight_7kg, 49)
# threshold_weight_8kg = np.percentile(emg_envelope_weight_8kg, 60)
# threshold_weight_10kg = np.percentile(emg_envelope_weight_10kg, 64)

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

############# Create arrays with the features ###########
all_mav_values = []
all_rms_values = []
all_wl_values = []
all_zc_values = []
all_windows = []

all_mav_values.extend(mav_values_mvc)
all_mav_values.extend(mav_values_weight)
all_mav_values.extend(mav_values_no_weight)
all_mav_values.extend(mav_values_weight_1kg)
all_mav_values.extend(mav_values_weight_3kg)
all_mav_values.extend(mav_values_weight_4kg)
all_mav_values.extend(mav_values_weight_7kg)
all_mav_values.extend(mav_values_weight_8kg)
all_mav_values.extend(mav_values_weight_10kg)

all_rms_values.extend(rms_values_mvc)
all_rms_values.extend(mav_values_weight)
all_rms_values.extend(mav_values_no_weight)
all_rms_values.extend(rms_values_weight_1kg)
all_rms_values.extend(rms_values_weight_3kg)
all_rms_values.extend(rms_values_weight_4kg)
all_rms_values.extend(rms_values_weight_7kg)
all_rms_values.extend(rms_values_weight_8kg)
all_rms_values.extend(rms_values_weight_10kg)

all_wl_values.extend(wl_values_mvc)
all_wl_values.extend(wl_values_weight)
all_wl_values.extend(wl_values_no_weight)
all_wl_values.extend(wl_values_weight_1kg)
all_wl_values.extend(wl_values_weight_3kg)
all_wl_values.extend(wl_values_weight_4kg)
all_wl_values.extend(wl_values_weight_7kg)
all_wl_values.extend(wl_values_weight_8kg)
all_wl_values.extend(wl_values_weight_10kg)

all_windows.extend(window_labels_mvc)
all_windows.extend(mav_values_weight)
all_windows.extend(mav_values_no_weight)
all_windows.extend(window_labels_weight_1kg)
all_windows.extend(window_labels_weight_3kg)
all_windows.extend(window_labels_weight_4kg)
all_windows.extend(window_labels_weight_7kg)
all_windows.extend(window_labels_weight_8kg)
all_windows.extend(window_labels_weight_10kg)

#SECTION 4: Data split, nomralization and organization for model fit


########################### SPLIT DATA ####################

#Run this part to prepare the data for input to the SVM model

#Assuming X_windows and Y_windows are your Python lists
X_windows = [all_mav_values, all_rms_values]
Y_windows =  all_windows

# Convert Python lists to NumPy arrays and transpose to match the shape
X_windows = np.array(X_windows).T
Y_windows = np.array([Y_windows]).T

print("Number of samples in X_windows:", X_windows.shape[0])
print("Number of samples in Y_windows:", Y_windows.shape[0])

# Convert Y_windows to integers
Y_windows = Y_windows.astype(int)

print("Unique values in Y_windows:", np.unique(Y_windows))


# Split the shuffled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_windows, Y_windows, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Change the shape of y to (n_samples,) using ravel()
y_train = y_train.ravel()
y_test = y_test.ravel()


print("Y_TEST", y_test)
print("y_train", y_train)



#Section: K-Nearest Neigbor algorithm

knn = KNeighborsClassifier(n_neighbors=3)  # Initialize KNN with desired number of neighbors
knn.fit(X_train, y_train)  # Train the KNN classifier


# Assuming X_test contains test features
y_pred = knn.predict(X_test)  # Predict classes for test features


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save the model to a file
dump(knn, 'knn_model.joblib')




############ GET STREAMING DATA FROM EMG SENSOR################

#Open serial port on sensor
exg1_reg_array = bytearray(10)

# Lists to store data for plotting
timestamps = []
chanal1_values = []
chanal2_values = []

# Initialize the sample rate and sample count
sample_rate = 1024  # Samples per second
sample_count = 0

if len(sys.argv) < 2:
    print("No device specified.")
else:
    ser = serial.Serial(sys.argv[1], 115200)
    ser.reset_input_buffer()
    print("Port open...")

    # send start streaming command
    ser.write(struct.pack('B', 0x07))
    print("starting...")
    ACK = ser.read(1)

    # # Initialize a figure for live plotting
    # plt.ion()
    # fig, ax = plt.subplots()
    # #line1, = ax.plot([], [], label='Channel 1')
    # line2, = ax.plot([], [], label='Channel 2')
    # ax.set_xlabel('Timestamp (s)')  # Updated the label to show seconds
    # ax.set_ylabel('Value (mV)')
    # ax.legend()
    # ax.set_title('Live Data Streaming')

   

    # read incoming data

    ddata = b""
    framesize = 11  # 1-byte packet type + 2-byte timestamp + 14-byte ExG data
    i = 0

    try:

        while True:
            ddata += ser.read(framesize)

            while len(ddata) >= framesize:
                data = ddata[0:framesize]
                ddata = ddata[framesize:]
                (packettype,) = struct.unpack('B', data[0:1])

                # Calculate the timestamp based on sample count and sample rate
                timestamp = sample_count / sample_rate
                sample_count += 1

                c1ch1_raw = struct.unpack('>i', (data[5:8] + b'\0'))[0] >> 8
                c1ch2_raw = struct.unpack('>i', (data[8:11] + b'\0'))[0] >> 8
                c1status = struct.unpack('B', data[4:5])[0]

                #Calculate mV
                c1ch1_mV = (c1ch1_raw * ((2.42 * 1000) / 12)) / (2 ** 23) - 1
                c1ch2_mV = (c1ch2_raw * ((2.42 * 1000) / 12)) / (2 ** 23) - 1

                # Append data for plotting
                timestamps.append(timestamp)
                chanal1_values.append(c1ch1_mV)
                chanal2_values.append(c1ch2_mV)
                i = i + 1

            if (i >= 500):

                # # Update the live plot: MAKES THE APPLICATION SLOWER
                # i = 0
                # line2.set_data(timestamps, chanal2_values)
                # ax.relim()
                # ax.autoscale_view()
                # fig.canvas.flush_events()

                # Process the current window
                
                # ACCdata_shimmer = ['timestamp', 'c1ch2_mV']  
                # raw_channel_data = ACCdata_shimmer['c1ch2_mV']
                # emg_correctmean_shimmer = emgf.remove_mean(raw_channel_data, ACCdata_shimmer['timestamp'])
                # emg_filtered_shimmer, emg_envelope_shimmer, emg_rectified_shimmer = emgf.alltogether(ACCdata_shimmer['timestamp'], emg_correctmean_shimmer)

                # Process the current window

                raw_channel_data = chanal2_values
                emg_correctmean_shimmer = emgf.remove_mean(raw_channel_data, timestamps)
                emg_filtered_shimmer, emg_envelope_shimmer, emg_rectified_shimmer = emgf.alltogether(timestamps, emg_correctmean_shimmer)


                threshold = np.percentile(emg_envelope_shimmer, 49.5)
                mav_values_shimmer, rms_values_shimmer, wl_values_shimmer, zc_values_shimmer, window_labels_shimmer = emgf.emg_td_features (emg_rectified_shimmer, 50, 10,threshold)



                #TEST THE MODEL WITH STREAMING DATA ONLINE:

                # Load the saved model
                loaded_model = load('knn_model.joblib')

                X_windows = [mav_values_shimmer, rms_values_shimmer] 
                Y_windows =  window_labels_shimmer

                # Combine the arrays into a 2D array
                X_windows_combined = np.column_stack(X_windows)

                #Transform the features using the fitted scaler
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                #X_train = loaded_model.fit_transform(X_train)

                X_windows_scaled = scaler.transform(X_windows_combined)

                Y_windows = np.array([Y_windows]).T


                # Assuming you have new data in X_new for predictions
                X_new= X_windows_scaled
                predictions = loaded_model.predict(X_new)


                #Send data to motor control

                # Find the indices where the values in Y_new_pred change
                change_indices = np.where(np.diff(predictions) != 0)[0]

                # Initialize an array to store the sequence of zeros and ones
                sequence_array = np.zeros(len(predictions), dtype=int)

                # Process predictions based on consecutive sequences
                for i in range(len(change_indices) + 1):
                    if i == 0:
                        start_index = 0
                    else:
                        start_index = change_indices[i - 1] + 1

                    if i == len(change_indices):
                        end_index = len(predictions)
                    else:
                        end_index = change_indices[i]

                        sequence = predictions[start_index:end_index + 1]
                        majority = np.argmax(np.bincount(sequence))

                        # Update the sequence_array with the majority value for the current sequence
                        sequence_array[start_index:end_index + 1] = majority

                    # Check every first 10 numbers and determine the majority for the entire array
                    for i in range(0, len(sequence_array), 10):
                        subset = sequence_array[i:i+10]
                        majority = np.argmax(np.bincount(subset))

                        numeric_array = np.zeros(len(sequence_array))
                                    
                        # Your code to send ones goes here           
                        numeric_array[i:i+10] = majority                                                        
                                # Analyze the processed window if needed
                        print("output_command :", str(majority))




    except KeyboardInterrupt:

        # send stop streaming command
        ser.write(struct.pack('B', 0x20))

        # close serial port
        ser.close()
        print()

        # Calculate the accuracy of the model
        accuracy = accuracy_score(Y_windows, predictions) *100
        print('Online Accuracy:', accuracy)

        print("All done!")

















