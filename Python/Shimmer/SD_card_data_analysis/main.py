import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import EMG_functions as emgf
import numpy as np

#Sampling frequency
sf = 1024

file = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session1\full_contraction_emg.txt'
column_names = ['time', 'emgch1']
ACCdata_mvc = pd.read_csv(file, names=column_names, sep='\s+', skiprows=30, skipfooter=30, engine='python') #Full contraction without waight 

file2 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session2\emg_no_weight.txt'
column_names = ['time', 'emgch1']
ACCdata_no_weight = pd.read_csv(file2, names=column_names, sep='\s+', skiprows=30, skipfooter=30, engine='python') #Full contraction without waight 

file3 = r'D:\MDU\5th_year\Exsoskeleton\Python\Shimmer\SD_card_data_analysis\shimmer_data_SD\Shimmer_reading\Session3\emg_with_weight.txt'
column_names = ['time', 'emgch1']
ACCdata_weight = pd.read_csv(file3, names=column_names, sep='\s+', skiprows=30, skipfooter=30, engine='python') #Full contraction without waight 

################ Execute remove_mean function ##########
emg_correctmean_mvc = emgf.remove_mean(ACCdata_mvc['emgch1'], ACCdata_mvc['time'] / 1000) 
emg_correctmean_no_weight = emgf.remove_mean(ACCdata_no_weight['emgch1'], ACCdata_no_weight['time'] / 1000) 
emg_correctmean_weight = emgf.remove_mean(ACCdata_weight['emgch1'], ACCdata_weight['time'] / 1000)

####### Plot the emg_correctmean of different signals #####
#view_correctmean_mvc= emgf.visualize_remove_mean(emg_correctmean_mvc,ACCdata_mvc['emgch1'], ACCdata_mvc['time'] / 1000) # Plot correctmean
#view_correctmean_no_weight= emgf.visualize_remove_mean(emg_correctmean_no_weight,ACCdata_no_weight['emgch1'], ACCdata_no_weight['time'] / 1000) # Plot correctmean
#view_correctmean_weight= emgf.visualize_remove_mean(emg_correctmean_weight,ACCdata_weight['emgch1'], ACCdata_weight['time'] / 1000) # Plot correctmean

################ Execute alltogether function ##########
########### This function has the flowing:emg_filtered, emg_envelope, emg_rectified, emg_TD_Fetures,emg_sigment and labels
emg_filtered_mvc, emg_envelope_mvc, emg_rectified_mvc, mav_values_mvc, rms_values_mvc, wl_values_mvc, zc_values_mvc = emgf.alltogether (ACCdata_mvc['time'] / 1000, emg_correctmean_mvc )# sfreq has the sampling frequency 
emg_filtered_no_weight, emg_envelope_no_weight, emg_rectified_no_weight, mav_values_no_weight, rms_values_no_weight, wl_values_no_weight, zc_values_no_weight= emgf.alltogether (ACCdata_no_weight['time'] / 1000,  emg_correctmean_no_weight)# sfreq has the sampling frequency 
emg_filtered_weight, emg_envelope_weight, emg_rectified_weight, mav_values_weight, rms_values_weight, wl_values_weight, zc_values_weightt= emgf.alltogether (ACCdata_weight['time'] / 1000, emg_correctmean_weight)# sfreq has the sampling frequency 



####### Plot the view_alltogether of different signals #####
#view_alltogether_mvc= emgf.visualize_alltogether(ACCdata_mvc['time'] / 1000, emg_correctmean_mvc, emg_rectified_mvc, emg_envelope_mvc)
#view_alltogether_no_weight= emgf.visualize_alltogether(ACCdata_no_weight['time'] / 1000, emg_correctmean_no_weight, emg_rectified_no_weight, emg_envelope_no_weight)
#view_alltogether_weight= emgf.visualize_alltogether(ACCdata_weight['time'] / 1000, emg_correctmean_weight, emg_rectified_weight, emg_envelope_weight)

######## Plot all emg_envelopes of different signals #####
time_axis = range(0, len(emg_envelope_no_weight))
emg_envelopes = [emg_envelope_mvc, emg_envelope_no_weight, emg_envelope_weight]
view_emg_envelopes=emgf.plot_emg_envelopes_common_time(time_axis, emg_envelopes)

######## I made this plot to know the time for each envelope############
# print("Length of time_axis:", len(time_axis))
# for i, envelope in enumerate(emg_envelopes):
#     print(f"Length of emg_envelope {i + 1}:", len(envelope))

#### Plot the MAV, RMS of different signals####
# view_mav_rms_mvc= emgf. visualize_mav_rms(mav_values_mvc, rms_values_mvc)
# view_mav_rms_no_weight= emgf. visualize_mav_rms(mav_values_no_weight, rms_values_no_weight)
# view_mav_weight= emgf. visualize_mav_rms(mav_values_weight, rms_values_weight)

#### Plot the WL_ZC of different signals####
# view_wl_zc_mvc= emgf.visualize_wl_zc( emg_filtered_mvc, wl_values_mvc, zc_values_mvc)
# view_wl_zc_no_weight= emgf.visualize_wl_zc(emg_filtered_no_weight, wl_values_no_weight, zc_values_no_weight)
# view_wl_zc_weight= emgf.visualize_wl_zc(emg_filtered_weight, wl_values_weight, zc_values_weight)



########### Normalization part ###############
reference_mav = mav_values_mvc # Use the MAV from the reference record (e.g., max contraction without weight)
# Normalize each record based on RMS envelope and reference MAV
normalized_emg_record2 = [(rms_values_no_weight / reference_mav)*100  for rms_values_no_weight, reference_mav in zip(rms_values_no_weight, reference_mav)]
# Assuming rms_values_weight and reference_mav have the same length
normalized_emg_record3 = [(rms_values_weight / reference_mav)*100 for rms_values_weight, reference_mav in zip(rms_values_weight, reference_mav)]

# ####### plot the normalized_emg of different signals####
# view_normalized_emg = emgf.plot_normalized_emg(ACCdata_mvc['time'] / 1000, [ normalized_emg_record2, normalized_emg_record3])




#FREQUENCY DOMAIN


# Calculate FFT of the EMG signal
frequencies, magnitude = emgf.calculate_fft(ACCdata_mvc['emgch1'], sf)


# Visualize the frequency domain
view_frequency_domain=emgf.visualize_frequency_domain(frequencies, magnitude, xlim=(0, 550), ylim=(0, 1500))



plt.show()
