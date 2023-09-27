#!/usr/bin/python3
import sys
import struct
import serial
import matplotlib.pyplot as plt

 

# Function to convert the gain setting to gain value
def convert_exg_gain_setting_to_value(setting):
    if setting == 0:
        return 6
    elif setting == 1:
        return 1
    elif setting == 2:
        return 2
    elif setting == 3:
        return 3
    elif setting == 4:
        return 4
    elif setting == 5:
        return 8
    elif setting == 6:
        return 12
    else:
        return -1  # -1 means an invalid value

 

# Declare a bytearray to store configuration or gain settings
exg1_reg_array = bytearray(10)

 

# Lists to store data for plotting
timestamps = []
chanal1_values = []
chanal2_values = []

 

# Initialize the sample rate and sample count
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

 

    # Initialize a figure for live plotting
    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='Channel 1')
    line2, = ax.plot([], [], label='Channel 2')
    ax.set_xlabel('Timestamp (s)')  # Updated the label to show seconds
    ax.set_ylabel('Value (mV)')
    ax.legend()
    ax.set_title('Live Data Streaming')

 

    # read incoming data
    ddata = b""
    framesize = 18  # 1-byte packet type + 2-byte timestamp + 14-byte ExG data

 

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

 

                c2status = struct.unpack('B', data[11:12])[0]
                c2ch1_raw = struct.unpack('>i', (data[12:15] + b'\0'))[0] >> 8
                c2ch2_raw = struct.unpack('>i', (data[15:18] + b'\0'))[0] >> 8

 

                # Calculate the gain based on exg1_reg_array and apply it
                gain = convert_exg_gain_setting_to_value((exg1_reg_array[3] >> 4) & 7)
                c1ch1_mV = (c1ch1_raw * ((2.42 * 1000) / gain)) / (2 ** 23 - 1)
                c1ch2_mV = (c1ch2_raw * ((2.42 * 1000) / gain)) / (2 ** 23 - 1)
                c2ch1_mV = (c2ch1_raw * ((2.42 * 1000) / gain)) / (2 ** 23 - 1)
                c2ch2_mV = (c2ch2_raw * ((2.42 * 1000) / gain)) / (2 ** 23 - 1)

 

                # Calculate channel differences
                chanal1 = c1ch1_mV - c1ch2_mV
                chanal2 = c2ch1_mV - c2ch2_mV

 

                # Append data for plotting
                timestamps.append(timestamp)
                chanal1_values.append(chanal1)
                chanal2_values.append(chanal2)

 

                # Update the live plot
                line1.set_data(timestamps, chanal1_values)
                line2.set_data(timestamps, chanal2_values)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.flush_events()

 
                # print("0x%02x,%.2f,\t0x%02x,%8.2f,%8.2f,\t0x%02x,%8.2f,%8.2f" % (
                #     packettype, timestamp, c1status, chanal1, c1ch2_mV, c2status, chanal2, c2ch2_mV))
               
                print("0x%02x,%.2f,%8.2f,%8.2f,\t0x%02x,%8.2f,%8.2f" % (
                    packettype, timestamp, chanal1, c1ch2_mV, c2status, chanal2, c2ch2_mV))

 

    except KeyboardInterrupt:
        # send stop streaming command
        ser.write(struct.pack('B', 0x20))
        # close serial port
        ser.close()
        print()
        print("All done!")

 

plt.ioff()
plt.show()