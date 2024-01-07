import sys
import struct
import serial
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import EMG_functions as emgf
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import seaborn as sns
import time

from MotorControllerBicepsonlySPEED import *
import threading

# #Sampling frequency
sf = 1024

# # 1 DATA COLLECTION

 #Training and testind data

 ######sos: CHANGE PATH OF TRAINING DATA TO YOUR OWN!

file2 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\emg_no_weight.txt'
column_names = ['time', 'emgch1']
ACCdata_no_weight = pd.read_csv(file2, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')

file = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\full_contraction_emg.txt'
column_names = ['time', 'emgch1']
ACCdata_mvc = pd.read_csv(file, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')

file3 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\emg_with_weight.txt'
column_names = ['time', 'emgch1']
ACCdata_weight = pd.read_csv(file3, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python') 

file4 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\Assessment1_Session1_Shimmer_9D70_Calibrated_PC.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_7kg = pd.read_csv(file4, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file5 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\Session2_7.5kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_8kg = pd.read_csv(file5, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file6 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\Session3_5kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_5kg = pd.read_csv(file6, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file7 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\Session4_2.5kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_3kg = pd.read_csv(file7, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file8 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\Session5_1.25kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_1kg = pd.read_csv(file8, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  

file9 = r'D:\MDH\Final Year\Online_Code_SVM\OfficialTEST_data\Session1_10kg.txt'
column_names = ['time', 'emgch1']
ACCdata_weight_10kg = pd.read_csv(file9, names=column_names, sep='\s+', skiprows=3000, skipfooter=3000, engine='python')  


#SECTION 2: Data filtering

################ Execute remove_mean function ##########
emg_correctmean_no_weight = emgf.remove_mean(ACCdata_no_weight['emgch1'], ACCdata_no_weight['time'] / 1000) 
emg_correctmean_mvc = emgf.remove_mean(ACCdata_mvc['emgch1'], ACCdata_mvc['time'] / 1000) 
emg_correctmean_weight = emgf.remove_mean(ACCdata_weight['emgch1'], ACCdata_weight['time'] / 1000)
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
emg_filtered_weight_1kg, emg_envelope_weight_1kg, emg_rectified_weight_1kg= emgf.alltogether (ACCdata_weight_1kg['time'] / 1000, emg_correctmean_weight_1kg)
emg_filtered_weight_3kg, emg_envelope_weight_3kg, emg_rectified_weight_3kg= emgf.alltogether (ACCdata_weight_3kg['time'] / 1000, emg_correctmean_weight_3kg)
emg_filtered_weight_4kg, emg_envelope_weight_4kg, emg_rectified_weight_4kg= emgf.alltogether (ACCdata_weight_5kg['time'] / 1000, emg_correctmean_weight_4kg)
emg_filtered_weight_7kg, emg_envelope_weight_7kg, emg_rectified_weight_7kg= emgf.alltogether (ACCdata_weight_7kg['time'] / 1000, emg_correctmean_weight_7kg)
emg_filtered_weight_8kg, emg_envelope_weight_8kg, emg_rectified_weight_8kg= emgf.alltogether (ACCdata_weight_8kg['time'] / 1000, emg_correctmean_weight_8kg)
emg_filtered_weight_10kg, emg_envelope_weight_10kg, emg_rectified_weight_10kg= emgf.alltogether (ACCdata_weight_10kg['time'] / 1000, emg_correctmean_weight_10kg)

#SECTION 3: Feature extraction
################## Features extraction USING emg_filtered and emg_rectified######################
# Threshold value for labeling
threshold = np.mean(emg_envelope_no_weight)

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

########################### SPLIT DATA ####################

# #Assuming X_windows and Y_windows are your Python lists
X_windows = [all_mav_values, all_rms_values]
Y_windows =  all_windows
# Convert Python lists to NumPy arrays and transpose to match the shape
X_windows = np.array(X_windows).T
Y_windows = np.array([Y_windows]).T
# print("Number of samples in X_windows:", X_windows.shape[0])
# print("Number of samples in Y_windows:", Y_windows.shape[0]) 
# Convert Y_windows to integers
Y_windows = Y_windows.astype(int)
# print("Unique values in Y_windows:", np.unique(Y_windows))
# Split the shuffled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_windows, Y_windows, test_size=0.2, random_state=42)
# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Change the shape of y to (n_samples,) using ravel() 1D array
y_train = y_train.ravel()
y_test = y_test.ravel()
# print("Y_TEST", y_test)
# print("y_train", y_train)

########################################## MACHINE LEARNING #############################

#######################  REMOVE SAVED MODEL/SVM MODEL ###################
# import os

# # Specify the filename of the model you want to delete
# model_to_delete_SVM = 'svm_model.pkl'

# # Check if the file exists before trying to delete it
# if os.path.exists(model_to_delete_SVM):
#     os.remove(model_to_delete_SVM)
#     print(f"Model {model_to_delete_SVM} has been deleted.")
# else:
#     print(f"Model {model_to_delete_SVM} does not exist, so nothing was deleted.")


# # Specify the filename of the model you want to delete
# model_to_delete2 = 'standard_scaler_SVM.pkl'

# # Check if the file exists before trying to delete it
# if os.path.exists(model_to_delete2):
#     os.remove(model_to_delete2)
#     print(f"Model {model_to_delete2} has been deleted.")
# else:
#     print(f"Model {model_to_delete2} does not exist, so nothing was deleted.")


# import os

# # Specify the filenames of the models you want to delete
# model_files = ['svm_model.joblib', 'standard_scaler_SVM.joblib']

# for model_file in model_files:
#     if os.path.exists(model_file):
#         os.remove(model_file)
#         print(f"Model {model_file} has been deleted.")
#     else:
#         print(f"Model {model_file} does not exist, so nothing was deleted.")


# ############################### SVM OF NEW DATA ###################

# #Create and train an SVM classifier
# svm_classifier = SVC(kernel='linear', C=1.0)
# svm_classifier.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = svm_classifier.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# accuracy_percentage = accuracy * 100
# report = classification_report(y_test, y_pred)

# print(f'Accuracy: {accuracy_percentage:.2f}%')
# print(report)

####################### Save / Load data ####################
from joblib import load,dump
##Save the model to a file
# dump(svm_classifier, 'svm_model.joblib')
# dump(scaler, 'standard_scaler_SVM.joblib')
scaler_filepath2= 'D:\MDH\Final Year\svm_model.joblib'
svm_model = load(scaler_filepath2)

scaler_filepath = 'D:\MDH\Final Year\online_SVM\SVM\standard_scaler_SVM.joblib'  # Update the file path
scaler = load(scaler_filepath)

################################# ONLINE STREAMING DATA ##########################
############################ GET STREAMING DATA FROM EMG SENSOR################

#Open serial port on sensor
exg1_reg_array = bytearray(10)

# Lists to store data for plotting
timestamps = []
chanal1_values = []
chanal2_values = []

# Initialize the sample rate and sample count
#sample_rate = 1024  # Samples per second
sample_rate = 512  # Samples per second
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
    
    motor_startup_torque()

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
                raw_channel_data = chanal2_values
                emg_correctmean_shimmer = emgf.remove_mean(raw_channel_data, timestamps)
                emg_filtered_shimmer, emg_envelope_shimmer, emg_rectified_shimmer = emgf.alltogether(timestamps, emg_correctmean_shimmer)

                threshold1 = np.percentile(emg_envelope_shimmer, 49.5)

                mav_values_shimmer, rms_values_shimmer, wl_values_shimmer, zc_values_shimmer, window_labels_shimmer = emgf.emg_td_features (emg_rectified_shimmer, 50, 10,threshold1)



             

                 ######################### SVM model on a stream data ############
                
                X_windows = [ mav_values_shimmer, rms_values_shimmer] 
                Y_windows =  window_labels_shimmer

                # Combine the arrays into a 2D array
                X_windows_combined = np.column_stack(X_windows)
                # Transform the features using the fitted scaler
                X_windows_scaled = scaler.transform(X_windows_combined)

                Y_windows = np.array([Y_windows]).T

                # Assuming you want to use the combined and scaled features
                Y_new_pred = svm_model.predict(X_windows_scaled)

                # Assuming Y_new_pred contains the predicted values for the new data
                Y_windows = Y_windows.ravel()  # Flatten the Y_windows array
                Y_new_pred = Y_new_pred.ravel()  # Flatten the Y_new_pred array

                #You can then evaluate the predictions and print the results
                accuracy = accuracy_score(Y_windows, Y_new_pred)
                accuracy_percentage = accuracy * 100
                report = classification_report(Y_windows, Y_new_pred)


                ###########################################################################
                # Assuming Y_new_pred contains the predictions from your SVM model
                # Define your segment size (adjust this as needed)
                
                segment_size = 3
                # Find the indices where the values in Y_new_pred change
                change_indices = np.where(np.diff(Y_new_pred) != 0)[0]

                # Initialize an array to store the sequence of zeros and ones
                sequence_array = np.zeros(len(Y_new_pred), dtype=int)

                # Find the indices where the values in Y_new_pred change
                change_indices = np.where(np.diff(Y_new_pred) != 0)[0]
 
                # Initialize an array to store the sequence of zeros and ones
                sequence_array = np.zeros(len(Y_new_pred), dtype=int)
 
                # Process predictions based on consecutive sequences
                for i in range(len(change_indices) + 1):
                    if i == 0:
                        start_index = 0
                    else:
                        start_index = change_indices[i - 1] + 1
 
                    if i == len(change_indices):
                        end_index = len(Y_new_pred)
                    else:
                        end_index = change_indices[i]
 
                    sequence = Y_new_pred[start_index:end_index + 1]
                    majority = np.argmax(np.bincount(sequence))
 
                    # Update the sequence_array with the majority value for the current sequence
                    sequence_array[start_index:end_index + 1] = majority
                    
                # Process the sequence array in segments++
                numeric_array = np.zeros(len(sequence_array))
                
                for i in range(0, len(sequence_array), segment_size):
                    subset = sequence_array[i:i + segment_size]
                    majority = np.argmax(np.bincount(subset))
 
                    # Update the numeric_array with the majority for the current segment
                    numeric_array[i:i + segment_size] = majority

                print("output_command :", str(majority))
                motor_controller(majority)
                                         

    except KeyboardInterrupt:

        # send stop streaming command
        ser.write(struct.pack('B', 0x20))

        # close serial port
        ser.close()
        print()

        # Calculate the accuracy of the model
        accuracy = accuracy_score(Y_windows, Y_new_pred) *100
        print('Online Accuracy:', accuracy)
        # #print("Distribution of classes in Y_new_pred:", np.bincount(Y_new_pred))
        # # Count occurrences of 1 in Y_new_pred
        count_ones = np.bincount(Y_new_pred)[1]
        count_Zeros = np.bincount(Y_new_pred)[0]
        
        print(f'Number of instances predicted as class 1: {count_ones}')
        print(f'Number of instances predicted as class 0: {count_Zeros}')

        # Count the number of 1s and 0s in numeric_array
        num_ones = np.count_nonzero(numeric_array == 1)
        num_zeros = np.count_nonzero(numeric_array == 0)
        print("Number of 1 From numeric_array:", num_ones)
        print("Number of 0 From numeric_array:", num_zeros)       

        print("All done!")






