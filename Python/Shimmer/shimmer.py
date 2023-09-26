#!/usr/bin/python3

import sys

import struct

import serial

 

def wait_for_ack():

    ddata = b""

    ack = struct.pack('B', 0xff)

    while ddata != ack:

        ddata = ser.read(1)

    return

 

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

 

if len(sys.argv) < 2:

    print("No device specified.")

else:

    ser = serial.Serial(sys.argv[1], 115200)

    ser.reset_input_buffer()

    print("Port open...")

 

    # send start streaming command

    ser.write(struct.pack('B', 0x07))

    wait_for_ack()

    print("Start sent...")

 

    # read incoming data

    ddata = b""

    numbytes = 0

    framesize = 18  # 1-byte packet type + 2-byte timestamp + 14-byte ExG data

 

    print("Packet Type,Timestamp,Chip1 Status,Chip1 Channel1,Chip1 Channel2,Chip2 Status,Chip2 Channel1,Chip2 Channel2")

    try:

        while True:

            while numbytes < framesize:

                ddata += ser.read(framesize)

                numbytes = len(ddata)

 

            data = ddata[0:framesize]

            ddata = ddata[framesize:]

            numbytes = len(ddata)

 

            (packettype,) = struct.unpack('B', data[0:1])

 

            (ts0, ts1, ts2, c1status) = struct.unpack('BBBB', data[1:5])

            timestamp = ts0 + ts1 * 256 + ts2 * 65536

 

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

 

            print("0x%02x,%06x,\t0x%02x,%8.2f,%8.2f,\t0x%02x,%8.2f,%8.2f" % (

                packettype, timestamp, c1status, c1ch1_mV, c1ch2_mV, c2status, c2ch1_mV, c2ch2_mV))

 

    except KeyboardInterrupt:

        # send stop streaming command

        ser.write(struct.pack('B', 0x20))

        wait_for_ack()

        # close serial port

        ser.close()

        print()

        print("All done!")