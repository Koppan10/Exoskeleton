#CODE DESCRIPTION:
#
#Purpose: Neural network model to detect muscle contraction 
#
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


#######################  REMOVE SAVED MODEL ###################
model = keras.models.Sequential

#DELETE PREVIOUS MODEL
# del model
# #FREE UP RESOURCES FOR NEST MODEL TRAINING
# keras.backend.clear_session()


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
# Calculate a lower threshold for detecting contractions (e.g., 70th percentile)
threshold = np.percentile(emg_envelope_no_weight, 55)  # Adjust percentile as needed

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



####Frequency domain features#####

# # Calculate FFT of the EMG signal
# frequencies, magnitude = emgf.calculate_fft(ACCdata_mvc['emgch1'], sf)


# # Visualize the frequency domain
# view_frequency_domain=emgf.visualize_frequency_domain(frequencies, magnitude, xlim=(0, 550), ylim=(0, 1500))


#Power spectra density (PDS) in frequency representation



#SECTION 4: Data split, nomralization and organization for model fit


########################### SPLIT DATA ###################

#Run this part to prepare the data for input to the ANN model

#Assuming X_windows and Y_windows are your Python lists
X_windows = [all_mav_values, all_rms_values, all_wl_values]
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


np.random.seed(42) #set random seeds in order to get the same result every 
#every time we run this model
tf.random.set_seed(42)

#print(features.shape) #3 columns and xxx records in the dataset for training
#print()


#SECTION 5: Model construction and train

# # # 7 MODEL TRAINING

#Struction of the classification neural network
#Single output neuron without any activation function
model = keras.models.Sequential([
    keras.layers.Dense(10, activation="relu", input_shape=X_train.shape[1:]), #X_train has a shape of (num_samples, num_features)
    keras.layers.Dropout(0.5),  # 0.5 is the dropout rate
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dropout(0.3),  # Adjust the dropout rate
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid") #sigmoid, softmax
])

# Visualize the model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#print(model.summary())


learning_rate=0.02

#cross entropy for loss function: sparse_categorical_crossentropy, binary_crossentropy, categorical_crossentropy
model.compile(loss="binary_crossentropy", 
              optimizer=keras.optimizers.SGD(lr=learning_rate),  #lr=learning rate
              metrics=['accuracy'])#optional parameter: mse for regression

model_history = model.fit(X_train, y_train,batch_size=128, epochs=40,validation_split=0.1)


# SECTION 7: Model Evaluation

# # # # # # 8 MODEL EVALUATION

# #model_history.params


model_history.history #values of metices in the form of a dictionaty
#plot this dictionary=> gives how training loss and validaion loss
#are changing with each epoch and whether we have achieved the convergence or not


pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()

print("OK")


# #If we run more epoches this will further deceases the losses and improve the 
# #accuracy of our model
# #One way to tell if you have achieved convergence or not,if you have to incrase you epoch 
# #value or not.
# #Improve the accuracy by increasing the epoches and rerun the model


#Evaluate the performance of the model on our test set
#Output here: loss, accuracy on test set
print(model.evaluate(X_test, y_test))

# # # # # 9 PREDICTION

# # #TO PREDICT THE VALUES ON THE NEW DATASET YOU CAN USE THIS: To predict values on a new dataset



# y_predicted=model.predict(X_test)
# print (y_predicted)
 
predicted_classes = model.predict(X_test).round().astype(int)
print(predicted_classes)

#SAVE MODEL 
model.save("ANN_regression")
# %pwd #get directory location of the model on your PC

#   #CHANNGE MODEL directory on PC
# # %cd C:\\Users\\Irini

# #Loading the model again
# model = keras.models.load_model("my_Func_model.h5")

print("Managed")

# ####### Plot the view_alltogether of different signals #####

# view_alltogether_mvc= emgf.visualize_alltogether(ACCdata_mvc['time'] / 1000, emg_correctmean_mvc, emg_rectified_mvc, emg_envelope_mvc, title="  MVC")
# view_alltogether_weight= emgf.visualize_alltogether(ACCdata_weight['time'] / 1000, emg_correctmean_weight, emg_rectified_weight, emg_envelope_weight, title="  Weight")
# view_alltogether_no_weight= emgf.visualize_alltogether(ACCdata_no_weight['time'] / 1000, emg_correctmean_no_weight, emg_rectified_no_weight, emg_envelope_no_weight,title="  No_weight")
# view_alltogether= emgf.visualize_alltogether(ACCdata_file7['time'] / 1000, emg_correctmean_file7, emg_rectified_f7, emg_envelope_f7,title="File7")


# time_axis = range(0, len(emg_envelope_no_weight))
# emg_envelopes = [emg_envelope_mvc, emg_envelope_no_weight, emg_envelope_weight]
# view_emg_envelopes=emgf.plot_emg_envelopes_common_time(time_axis, emg_envelopes)

plt.show()






















# # Find the indices where the values in Y_new_pred change
# change_indices = np.where(np.diff(Y_new_pred) != 0)[0]
 
# # Initialize an array to store the sequence of zeros and ones
# sequence_array = np.zeros(len(Y_new_pred), dtype=int)
 
# # Process predictions based on consecutive sequences
# for i in range(len(change_indices) + 1):
#     if i == 0:
#         start_index = 0
#     else:
#         start_index = change_indices[i - 1] + 1
 
#     if i == len(change_indices):
#         end_index = len(Y_new_pred)
#     else:
#         end_index = change_indices[i]
 
#     sequence = Y_new_pred[start_index:end_index + 1]
#     majority = np.argmax(np.bincount(sequence))
 
#     # Update the sequence_array with the majority value for the current sequence
#     sequence_array[start_index:end_index + 1] = majority
 
#     # Print the start and end indices of each sequence
#     #print(f"Sequence {start_index+1}-{end_index+1} is labeled as {majority}")
 
# # Print the final sequence_array
# #print("Sequence Array:", sequence_array)
 
# numeric_array = np.zeros(len(sequence_array))
# # Check every first 10 numbers and determine the majority for the entire array
# for i in range(0, len(sequence_array), 10):
#     subset = sequence_array[i:i+10]
#     majority = np.argmax(np.bincount(subset))
   
#     # Send either ones or zeros based on the majority
#     if majority == 0:
#         print(f"Sending zeros for sequence {i+1}-{i+10}")
#         # Your code to send zeros goes here
#     else:
#         print(f"Sending ones for sequence {i+1}-{i+10}")
#         # Your code to send ones goes here
   
#     numeric_array[i:i+10] = majority
 
# print("Numeric Array:", numeric_array)
 






