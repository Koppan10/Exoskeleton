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
   framesize = 18  # 1byte packet type + 2byte timestamp + 14byte ExG data

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

         # (timestamp, c1status) = struct.unpack('HB', data[1:4])
         (ts0, ts1, ts2, c1status) = struct.unpack('BBBB', data[1:5])
         timestamp = ts0 + ts1 * 256 + ts2 * 65536
         # 24-bit signed values MSB values are tricky, as struct only supports 16-bit or 32-bit
         # pad with zeroes at LSB end and then shift the result

         c1ch1 = struct.unpack('>i', (data[5:8] + b'\0'))[0] >> 8
         c1ch2 = struct.unpack('>i', (data[8:11] + b'\0'))[0] >> 8
        # (c2status,) = struct.unpack('B', data[11])
         # Change this line:
         # (c2status,) = struct.unpack('B', data[11])
            # To this line:
         c2status = struct.unpack('B', data[11:12])[0]
         c2ch1 = struct.unpack('>i', (data[12:15] + b'\0'))[0] >> 8
         c2ch2 = struct.unpack('>i', (data[15:18] + b'\0'))[0] >> 8
         print("0x%02x,%06x,\t0x%02x,%8d,%8d,\t0x%02x,%8d,%8d" % (
         packettype, timestamp, c1status, c1ch1, c1ch2, c2status, c2ch1, c2ch2))

         #Convert ddata to mV
         # Physical Quantity (in mV) = (Raw Data * Scaling Factor) + Offset

   except KeyboardInterrupt:
      # send stop streaming command
      ser.write(struct.pack('B', 0x20))
      wait_for_ack()
      # close serial port
      ser.close()
      print()
      print("All done!")


#####################################################################################
'''

try:
    while True:
        while numbytes < framesize:
            ddata += ser.read(framesize)
            numbytes = len(ddata)

        data = ddata[0:framesize]
        ddata = ddata[framesize:]
        numbytes = len(ddata)

        (packettype,) = struct.unpack('B', data[0:1])

        # (timestamp, c1status) = struct.unpack('HB', data[1:4])
        (ts0, ts1, ts2, c1status) = struct.unpack('BBBB', data[1:5])
        timestamp = ts0 + ts1 * 256 + ts2 * 65536

        # Convert raw data to millivolts
        c1ch1_raw = struct.unpack('>i', (data[5:8] + b'\0'))[0] >> 8
        c1ch2_raw = struct.unpack('>i', (data[8:11] + b'\0'))[0] >> 8

        c1ch1_mV = (c1ch1_raw * scaling_factor_c1ch1) + offset_c1ch1
        c1ch2_mV = (c1ch2_raw * scaling_factor_c1ch2) + offset_c1ch2

        (c2status,) = struct.unpack('B', data[11])

        c2ch1_raw = struct.unpack('>i', (data[12:15] + b'\0'))[0] >> 8
        c2ch2_raw = struct.unpack('>i', (data[15:18] + b'\0'))[0] >> 8

        # You can add similar conversions for other channels if needed.

        print("Packet Type: 0x%02x, Timestamp: %06x" % (packettype, timestamp))
        print("Channel 1 (mV):", c1ch1_mV)
        print("Channel 2 (mV):", c1ch2_mV)
        print("Channel 1 Status: 0x%02x, Channel 2 Status: 0x%02x" % (c1status, c2status))
'''
        