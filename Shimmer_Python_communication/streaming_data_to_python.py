#This python script provides a communication between Shimmer sensor PNLY one channel signal. 
#We can stream  raw sensor data of one channel directly and view the muscle contraction including artifact, the RAW EMG signal.

import sys
import struct
import serial 
import matplotlib.pyplot as plt


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

    # Initialize a figure for live plotting
    plt.ion()
    fig, ax = plt.subplots()
    #line1, = ax.plot([], [], label='Channel 1')
    line2, = ax.plot([], [], label='Channel 2')
    ax.set_xlabel('Timestamp (s)')  # Updated the label to show seconds
    ax.set_ylabel('Value (mV)')
    ax.legend()
    ax.set_title('Live Data Streaming')

   

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

                #c2status = struct.unpack('B', data[11:12])[0]
                #c2ch1_raw = struct.unpack('>i', (data[12:15] + b'\0'))[0] >> 8
                #c2ch2_raw = struct.unpack('>i', (data[15:18] + b'\0'))[0] >> 8

               

                #Calculate mV
                c1ch1_mV = (c1ch1_raw * ((2.42 * 1000) / 12)) / (2 ** 23) - 1
                c1ch2_mV = (c1ch2_raw * ((2.42 * 1000) / 12)) / (2 ** 23) - 1

                #c2ch1_mV = (c2ch1_raw * ((2.42 * 1000) / gain)) / (2 ** 23 - 1)
                #c2ch2_mV = (c2ch2_raw * ((2.42 * 1000) / gain)) / (2 ** 23 - 1)

                # Append data for plotting
                timestamps.append(timestamp)
                chanal1_values.append(c1ch1_mV)
                chanal2_values.append(c1ch2_mV)
                i = i + 1

                if (i >= 500):

                # Update the live plot
                    i = 0
                    #line1.set_data(timestamps, chanal1_values)
                    line2.set_data(timestamps, chanal2_values)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.flush_events()

                #print("0x%02x,%.2f,%8.2f,%8.2f,%8.2f,%8d,%8d" % (packettype, timestamp, c1ch1_mV, c1ch2_mV,c1status,c1ch1_raw,c1ch2_raw))
                print("%.2f, %8.2f" % (timestamp, c1ch2_mV))
              




    except KeyboardInterrupt:

        # send stop streaming command
        ser.write(struct.pack('B', 0x20))

        # close serial port
        ser.close()
        print()
        print("All done!")

 
#plt.ioff()
#plt.show()