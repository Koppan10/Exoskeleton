import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal as sp_signal


def remove_mean(emg, time):
    # process EMG signal: remove mean
    emg_correctmean = emg - np.mean(emg)
    return emg_correctmean

def visualize_remove_mean(emg_correctmean,emg,time):
    #plot comparison of EMG with offset vs mean-corrected values
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Mean offset present')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Mean-corrected values')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    fig.tight_layout()
    fig_name = 'fig2.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name) 


def emg_filter(emg_correctmean,time):
    # create bandpass filter for EMG
    high = 20/(1000/2)
    low = 450/(1000/2)
    b, a = sp_signal.butter(4, [high,low], btype='bandpass')

    # process EMG signal: filter EMG
    emg_filtered = sp_signal.filtfilt(b, a, emg_correctmean)

    # plot comparison of unfiltered vs filtered mean-corrected EMG
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unfiltered EMG')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
   # plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Filtered EMG')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
   # plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    fig.tight_layout()
    fig_name = 'fig3.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)
    return emg_filtered

def emg_rectify(emg_filtered, time):
   # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # plot comparison of unrectified vs rectified EMG
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Unrectified EMG')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title('Rectified EMG')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
   # plt.ylim(-1.5, 1.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')

    fig.tight_layout()
    fig_name = 'fig4.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)
    return emg_rectified




def alltogether(time, emg, window_size=100, overlap=50, low_pass=2, sfreq=1024, high_band=20, low_band=450 ):
   
    """
    time: Time data
    emg: EMG data
    window_size: Size of the analysis window in samples
    overlap: Number of samples to overlap between windows (set to 0 for non-overlapping)
    low_pass: Low-pass cutoff frequency for envelope calculation
    sfreq: Sampling frequency
    high_band: High-pass cutoff frequency for filtering
    low_band: Low-pass cutoff frequency for filtering
    relaxation_duration: Duration of the initial relaxation phase in seconds
    flexing_duration: Duration of the flexing phase in seconds
    """

    # Normalizing cutoff frequencies to sampling frequency
    high_band = high_band / (sfreq / 2)
    low_band = low_band / (sfreq / 2)

    # Create bandpass filter for EMG
    b1, a1 = sp_signal.butter(4, [high_band, low_band], btype='bandpass')

    # Process EMG signal: filter EMG
    emg_filtered = sp_signal.filtfilt(b1, a1, emg)

    # Initialize lists to store feature values
    mav_values = []
    rms_values = []
    wl_values = []  # List to store Waveform Length
    zc_values = []  # List to store Zero Crossing

    # Process EMG signal: rectify and label windows
    emg_rectified = np.abs(emg_filtered)

    # Create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass / (sfreq / 2)
    b2, a2 = sp_signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp_signal.filtfilt(b2, a2, emg_rectified)

    # Calculate features using overlapping sliding windows
    step_size = window_size - overlap
    for i in range(0, len(emg_filtered) - window_size + 1, step_size):
        window = emg_rectified[i:i+window_size]
        mav = np.mean(window)
        rms = np.sqrt(np.mean(window**2))
        # Calculate Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(window)))
        # Calculate Zero Crossing (ZC)
        zc = np.sum(np.abs(np.diff(np.sign(window))))

        mav_values.append(mav)
        rms_values.append(rms)
        wl_values.append(wl)
        zc_values.append(zc)


    return emg_filtered, emg_envelope, emg_rectified, mav_values, rms_values, wl_values, zc_values


def plot_emg_envelopes_common_time(time_axis, envelopes):
  
    # Find the minimum length among the envelope arrays
    min_length = min(len(envelope) for envelope in envelopes)
    
    # Truncate all envelope arrays to match the minimum length
    envelopes = [envelope[:min_length] for envelope in envelopes]

    # Plot the truncated envelopes on the common time axis
    plt.figure(figsize=(10, 6))
    for i, envelope in enumerate(envelopes):
        plt.plot(time_axis[:min_length], envelope, label=f'EMG Envelope {i + 1}')

    plt.xlabel('Time (s)')
    plt.ylabel('EMG Envelope')
    plt.legend()
    plt.show()


def visualize_alltogether(time, emg, emg_rectified, emg_envelope):
    low_pass=10
    sfreq=1024
    high_band=20
    low_band=450
    # plot graphs
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.subplot(1, 3, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    
    plt.subplot(1, 3, 2)
    plt.subplot(1, 3, 2).set_title('Filtered,' + '\n' + 'rectified EMG: ' + str(int(high_band*sfreq)) + '-' + str(int(low_band*sfreq)) + 'Hz')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')

    plt.subplot(1, 3, 3)
    plt.subplot(1, 3, 3).set_title('Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass*sfreq)) + ' Hz')
    plt.plot(time, emg_envelope)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    
    fig_name = 'fig_' + str(int(low_pass*sfreq)) + '.png'
    fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name)

def visualize_mav_rms(mav_values, rms_values):
    
    time_axis = range(0, len(mav_values))

    # Plot MAV values
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, mav_values, label='MAV')
    plt.title('Mean Absolute Value (MAV)')
    plt.xlabel('Window Index')
    plt.ylabel('MAV Value')
    plt.grid(True)

    # Plot RMS values
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, rms_values, label='RMS', color='orange')
    plt.title('Root Mean Square (RMS)')
    plt.xlabel('Window Index')
    plt.ylabel('RMS Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_wl_zc(signal, wl_values, zc_values):
    time_axis = range(0, len(wl_values))

    # Ensure that the signal has the same time duration as the features
    signal = signal[:len(time_axis)]

    # Plot WL values
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, wl_values, label='Waveform Length (WL)')
    plt.title('Waveform Length (WL)')
    plt.xlabel('Window Index')
    plt.ylabel('WL Value')
    plt.grid(True)

    # Plot ZC values and the original signal
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, signal, label='Signal', color='blue')
    plt.plot(time_axis, zc_values, label='Zero Crossings (ZC)', color='orange')
    plt.title('Zero Crossings (ZC) and Signal')
    plt.xlabel('Window Index')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_normalized_emg(time_axis, normalized_emg_records):
   
    plt.figure(figsize=(10, 5))
    for i, emg_record in enumerate(normalized_emg_records):
        # Ensure that the time_axis matches the length of the current emg_record
        current_time_axis = time_axis[:len(emg_record)]
        plt.plot(current_time_axis, emg_record, label=f'Normalized EMG Record {i + 2}')


    plt.title('Normalized EMG Signals')
    plt.xlabel('Time (sec)')
    plt.ylabel('Normalized Value (%)')
    plt.legend()
    plt.grid(True)
    plt.show()



#FREQUENCY DOMAIN

def calculate_fft(emg_signal, sampling_rate):
    """
    Calculate the FFT of an EMG signal.

    Parameters:
        emg_signal (array): EMG signal data.
        sampling_rate (float): Sampling rate of the signal.

    Returns:
        tuple: Frequencies and magnitude of the FFT.

        Frequency analysis: 
 
        We want to know the frequency content of the RAW EMG signal to begin with.
        This will provide with the cut-off frequencies needed to create a correct 
        Band pass filter. 

        We can also check the freqiency domain of filtered data as well.
    """
    n = len(emg_signal)
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)
    fft_values = np.fft.fft(emg_signal)
    magnitude = np.abs(fft_values)
    return frequencies, magnitude



def visualize_frequency_domain(frequencies, magnitude, xlim=None, ylim=None):
    """
    Visualize the frequency domain (FFT) of an EMG signal.

    Parameters:
        frequencies (array): Frequencies from the FFT.
        magnitude (array): Magnitude from the FFT.
        xlim (tuple): Optional x-axis limits (e.g., (0, 500)).

    Returns:
        None (displays the plot).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')


    if xlim:
        plt.xlim(xlim)
    
    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.show()

