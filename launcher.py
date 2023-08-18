
#  Copyright (c) 2003-2022 Movella Technologies B.V. or subsidiaries worldwide.
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#  
#  1.	Redistributions of source code must retain the above copyright notice,
#  	this list of conditions and the following disclaimer.
#  
#  2.	Redistributions in binary form must reproduce the above copyright notice,
#  	this list of conditions and the following disclaimer in the documentation
#  	and/or other materials provided with the distribution.
#  
#  3.	Neither the names of the copyright holders nor the names of their contributors
#  	may be used to endorse or promote products derived from this software without
#  	specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
#  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
#  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  

# Requires installation of the correct Xsens DOT PC SDK wheel through pip
# For example, for Python 3.9 on Windows 64 bit run the following command
# pip install xsensdot_pc_sdk-202x.x.x-cp39-none-win_amd64.whl

import time, threading
from pynput import keyboard
from threading import Lock
from Sources.user_settings import *
from collections import defaultdict
from Sources import CallBackHandler as cbh
from Sources import client
import os, dataProcess
import xsensdot_pc_sdk
from Sources.config import server_ip, client_ip, main_port
from xsensdot_pc_sdk import XsPayloadMode_HighFidelitywMag, XsPayloadMode_ExtendedQuaternion, XsPayloadMode_CompleteQuaternion, XsPayloadMode_OrientationEuler, XsPayloadMode_OrientationQuaternion, XsPayloadMode_FreeAcceleration, XsPayloadMode_ExtendedEuler, XsPayloadMode_CompleteEuler, XsPayloadMode_HighFidelity, XsPayloadMode_DeltaQuantitieswMag, XsPayloadMode_DeltaQuantities, XsPayloadMode_RateQuantitieswMag, XsPayloadMode_RateQuantities, XsPayloadMode_CustomMode1, XsPayloadMode_CustomMode2, XsPayloadMode_CustomMode3, XsPayloadMode_CustomMode4, XsPayloadMode_CustomMode5

itr = 0  #.......unused variable
waitForConnections = True # ........... unused variable
totalDots = 5
MODES = {"HighFidelitywMag" : ("SampleTimeFine", "dq", "dv", "Angular velocity", "Acceleration", "Magnetic field", "Status"),
         "ExtendedQuaternion" : ("SampleTimeFine", "Orientation (Quaternion)", "Free acceleration", "Status"),
         "CompleteQuaternion" : ("SampleTimeFine", "Orientation (Quaternion)", "Free acceleration"),
         "OrientationEuler" : ("SampleTimeFine", "Orientation (Euler angles)"),
         "OrientationQuaternion" : ("SampleTimeFine", "Orientation (Quaternion)"),
         "FreeAcceleration": ("SampleTimeFine", "Free acceleration"),
         "ExtendedEuler" : ("SampleTimeFine", "Orientation (Euler angles)", "Free acceleration", "Status"),
         "CompleteEuler" : ("SampleTimeFine", "Orientation (Euler angles)", "Free acceleration"),
         "HighFidelity" : ("SampleTimeFine", "dq", "dv", "Angular velocity", "Acceleration", "Status"),
         "DeltaQuantitieswMag" : ("SampleTimeFine", "dq", "dv","Magnetic field"),
         "DeltaQuantities" : ("SampleTimeFine", "dq", "dv"),
         "RateQuantitieswMag" : ("SampleTimeFine", "Angular velocity", "Acceleration", "Magnetic field"),
         "RateQuantities" : ("SampleTimeFine", "Angular velocity", "Acceleration"),
         "CustomMode1" : ("SampleTimeFine", "Orientation (Euler angles)", "Free acceleration", "Angular velocity"),
         "CustomMode2" : ("SampleTimeFine", "Orientation (Euler angles)", "Free acceleration", "Magnetic field"),
         "CustomMode3" : ("SampleTimeFine", "Orientation (Quaternion)", "Angular velocity", ),
         "CustomMode4" : ("SampleTimeFine", "Orientation (Quaternion)",  "dq", "dv", "Angular velocity", "Acceleration", "Magnetic field", "Status"),
         "CustomMode5" : ("SampleTimeFine", "Orientation (Quaternion)", "Angular velocity", "Acceleration"),
         }
"""
All the modes available for the DOTs and their functions are provided above.
Remember to modify the 'from ... import' code at the front if MODE.keys() is changed!
"""
#
defaultMode = 'CustomMode1'
dataListLen = 5
varListLen = 5

def on_press(key):
    global waitForConnections
    waitForConnections = False

def timeLimit(order): 
    print(f"\nPlease input time limit for {order}. Unlimitation is taken as default.")
    if order == "reading data":
        print("Enter 'q' to terminate and quit.")
    while True:
        userIn = input()
        if userIn == '':
            return 0
        elif userIn.isdigit():
            return int(userIn)
        elif userIn == 'q' and order == "reading data":
            return "forcequit"
        else:
            print("Invalid input. Please retry.")


# Select the output mode of DOTs
def modeSelection():
    global defaultMode
    defaultNo = 14
    print("\n------------------------------")
    print("Please Select the output mode by input the ordinal number:")
    modes_List = list(MODES.items())
    optionsNum = len(modes_List)  # Numbers of mode options available
    for i in range(optionsNum):
        funcs = ''
        for func in modes_List[i][1]:
            funcs += f" {func}, "
        print(str(i + 1) + ". \033[34m" + modes_List[i][0] + "\033[0m : includes " + funcs, flush=True)
        time.sleep(0.05)
    print(f"{optionsNum} modes in total.\n Mode {defaultNo} {defaultMode} is taken as default.")
    print("------------------------------\n")

    while True:
        m = input(">>",)
        if m == '':
            print(f"Mode {defaultMode} selected.\n")
            return defaultMode
        elif m.isdigit():
            if 0 < int(m) < optionsNum + 1:
                mode = modes_List[int(m) - 1][0]
                print(f"Mode {mode} selected.\n")
                return mode
        else:
            print("Invalid input. Please retry.")

def winM(win, s):  #............unused function
    global itr
    while True:
        if itr % 100 == 0:
            print(itr)
            win.set_data(f"{itr}, {s}")

def quitting():
    print("\n------------------------------------")
    print("\nData Monitoring interrupted. What would you like to do next?")
    print("1. Terminate program and quit.")
    print("2. Redetermine the time limit and reselect the data for monitoring.")
    print("------------------------------------\n")

    while True:
        uin = input()
        if uin in ['1', '2']:
            return uin
        print("Invalid input. Please retry.")

if __name__ == "__main__":
    global deviceList
    global manager
    # Print SDK version
    version = xsensdot_pc_sdk.XsVersion()
    xsensdot_pc_sdk.xsdotsdkDllVersion(version)  # --------------- meaningless ???
    print(f"Using Xsens DOT SDK version: {version.toXsString()}")  # -------------- Visible

    # Create connection manager
    manager = xsensdot_pc_sdk.XsDotConnectionManager()
    if manager is None:
        print("Manager could not be constructed, exiting.")
        exit(-1)

    # Create and attach callback handler to connection manager
    callback = cbh.CallbackHandler()
    manager.addXsDotCallbackHandler(callback)

    # Start a scan and wait until we have found one or more DOT Devices
    print("Scanning for devices...")
    print(f"\033[34mAbout to find {totalDots} devices in total.\033[0m Scanning will be stopped automatically as soon as all the devices are found. Please revise this value at the front of the script if needed.")
    manager.enableDeviceDetection()

    # set a time limit for scanning
    limitation = timeLimit("scanning device")

    # Setup the keyboard input listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    time.sleep(0.5)
    print(f"Press any key {(lambda limitation: '' if limitation == 0 else f'or wait {limitation} seconds')(limitation)} to stop scanning...")
    connectedDOTCount = 0
    startTime = xsensdot_pc_sdk.XsTimeStamp_nowMs()

    while waitForConnections and not callback.errorReceived() and (lambda x: True if x == 0 else xsensdot_pc_sdk.XsTimeStamp_nowMs() - startTime <= x * 1000)(limitation):
        time.sleep(0.1)

        nextCount = len(callback.getDetectedDots())
        if nextCount != connectedDOTCount:
            print(f"Number of connected DOTs: {nextCount}. Press any key to start.")
            connectedDOTCount = nextCount
        if totalDots == nextCount: break

    manager.disableDeviceDetection()   #  -----------------  stop searching for new devices
    print("Stopped scanning for devices.")
    listener.stop()

    if len(callback.getDetectedDots()) == 0:
        print("No Xsens DOT device(s) found. Aborting.")
        exit(-1)

    mode = modeSelection()

    # Set the device tag name of a device
    deviceList = list()
    sequence_number = 0
    for portInfo in callback.getDetectedDots():  #  -------  callback.getDetectedDots() includs all the connected devices info
        address = portInfo.bluetoothAddress()
        # Go through all the devices to check if it is connection available

        print(f"Opening DOT with address: @ {address}")
        if not manager.openPort(portInfo):
            print(f"Connection to Device {address} failed, retrying...")
            print(f"Device {address} retry connected:")
            if not manager.openPort(portInfo):
                print(f"Could not open DOT. Reason: {manager.lastResultText()}")
                continue

        device = manager.device(portInfo.deviceId())
        if device is None:
            continue

        deviceList.append(device)
        print(f"Found a device with Tag: {device.deviceTagName()} @ address: {address}")

        filterProfiles = device.getAvailableFilterProfiles()  #  ------- Bluetooth settings
        print("Available filter profiles:")
        for f in filterProfiles:
            print(f.label())

        print(f"Current profile: {device.onboardFilterProfile().label()}")
        if device.setOnboardFilterProfile("General"):
            print("Successfully set profile to General")
        else:
            print("Setting filter profile failed!")

        with open(".\\rec.txt", '+w'):
            pass
        # ----------- Output logfile settings
        print("Setting quaternion CSV output")
        device.setLogOptions(xsensdot_pc_sdk.XsLogOptions_Quaternion)

        logFileName = "logfile_" + portInfo.bluetoothAddress().replace(':', '-') + ".csv"  #  + str(time.ctime()).replace(':', '-')
        print(f"Enable logging to: {logFileName}")

        # -----------------------------  enabling logging
        if not device.enableLogging(logFileName):  
            print(f"Failed to enable logging. Reason: {manager.lastResultText()}")
        print("Putting device into measurement mode.")

        # ------------------- setting modes
        modefunc = globals()[f"XsPayloadMode_{mode}"]
        if not device.startMeasurement(modefunc): 
            print(f"Could not put device into measurement mode {mode}. Reason: {manager.lastResultText()}")
            continue

    print("\nMain loop. Recording data for 10 seconds.")
    print("-----------------------------------------")

    # First printing some headers so we see which data belongs to which device
    s = ""
    for device in deviceList:
        s += f"{device.portInfo().bluetoothAddress():42}"
    print("%s" % s, flush=True)

    while True:
        dataTimeLimit = timeLimit("reading data")
        if dataTimeLimit == "forcequit": break
        time.sleep(0.5)
        time.sleep(0.5)
        print("Terminate and quit the program and restart the BLE service if the data below remain unchanged.")
        print("\033[34m\nPress key Ctrl^C to terminate.\033[0m")
        orientationResetDone = False
        startTime = xsensdot_pc_sdk.XsTimeStamp_nowMs()
        clnt = client.clnt_sckt((client_ip, main_port), (server_ip, main_port))
        dvs = []
        for ord, device in enumerate(deviceList):
            clnt.snd(create = ord, mode = mode)

        try:
            pkg_count = 0
            def mon(clnt):
                global callback, orientationResetDone, deviceList, startTime, dataTimeLimit, dvs, pkg_count
                itr = 0
                while (lambda x: True if x == 0 else xsensdot_pc_sdk.XsTimeStamp_nowMs() - startTime <= x * 1000)(dataTimeLimit):
                    itr += 1
                    if callback.packetsAvailable():
                        if not orientationResetDone:  # and xsensdot_pc_sdk.XsTimeStamp_nowMs() - startTime > 5000:
                            for device in deviceList:
                                print(f"\nResetting heading for device {device.portInfo().bluetoothAddress()}: ", end="", flush=True)
                                if device.resetOrientation(xsensdot_pc_sdk.XRM_Heading):
                                    print("OK", end="", flush=True)
                                else:
                                    print(f"NOK: {device.lastResultText()}", end="", flush=True)
                            print("\n", end="", flush=True)
                            orientationResetDone = True

                        for ord, device in enumerate(deviceList):
                            ex, ey, ez, ax, ay, az, vx, vy, vz, fax, fay, faz = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            # Retrieve a packet
                            packet = callback.getNextPacket(device.portInfo().bluetoothAddress())
                            if packet.containsOrientation():
                                euler = packet.orientationEuler()
                                ex, ey, ez = euler.x(), euler.y(), euler.z()
                            if packet.containsCalibratedAcceleration():
                                acce = packet.calibratedAcceleration()
                                ax, ay, az = acce[0], acce[1], acce[2]
                                #dataProcess.acceleration(acc.x(), acc.y(), acc.z())
                            if packet.containsVelocityIncrement():
                                dv = packet.velocityIncrement()
                                vx, vy, vz = dv[0], dv[1], dv[2]
                            if packet.containsCalibratedGyroscopeData():
                                pass
                                # gyr = packet.calibratedGyroscopeData()
                                # s += f"Gyr: {gyr}| "
                            if packet.containsFreeAcceleration():
                                Facc = packet.freeAcceleration()
                                fax, fay, faz = Facc[0], Facc[1], Facc[2]
                            # s += dvs[ord].run(ex = ex, ey = ey, ez = ez, ax = ax, ay = ay, az = az, vx = vx, vy = vy, vz = vz, fax = fax, fay = fay, faz = faz) + '\n'
                        # win.set_data(s)
                            clnt.snd(device = ord, t = xsensdot_pc_sdk.XsTimeStamp_nowMs(), dataListLen = dataListLen, varListLen = varListLen, ex = ex, ey = ey, ez = ez, ax = ax, ay = ay, az = az, vx = vx, vy = vy, vz = vz, fax = fax, fay = fay, faz = faz)
                            print(f"\r{pkg_count}", end = '', flush=True)
                            pkg_count += 1
                        # time.sleep(dataProcess.freq) # ---------- Set the frequency to 100Hz for computation convenience.
                return
            mon(clnt)
            '''
            win = dataProcess.RealtimeWindow('Monitor', dvs)
            winFrame = threading.Thread(target=mon, args=(clnt, ))
            winFrame.start()
            win.run()
            '''
        except KeyboardInterrupt or RuntimeError:
            r = quitting()
            if r == '1':
                print("\nTerminating...")
                break
            elif r == '2':
                print("\nrestarting...\n")
                clnt.terminate_sckt()
                time.sleep(0.5)
            
    print("\n-----------------------------------------", end="\n", flush=True)
    time.sleep(0.5)

    # Termination process
    try:
        clnt.terminate_sckt()
        print("Terminating devices...Please press key \033[31mCtrl^C\033[0m if timeout.")
        print("WARNING : There might be possibilities that this program could not be terminated by any means(even Ctrl^C is pressed). Please directly close the window and restart BLE service if that happens.")
        for device in deviceList:
            print(f"\nResetting heading to default for device {device.portInfo().bluetoothAddress()}: ", end="", flush=True)
            if device.resetOrientation(xsensdot_pc_sdk.XRM_DefaultAlignment):
                print("OK", end="", flush=True)
            else:
                print(f"NOK: {device.lastResultText()}", end="", flush=True)
        print("\n", end="", flush=True)
        print("\nStopping measurement...")
        for device in deviceList:
            if not device.stopMeasurement():
                print("Failed to stop measurement.")
            if not device.disableLogging():
                print("Failed to disable logging.")
    except KeyboardInterrupt:
        print("Device termination failed. Force quitting...")

    print("Closing ports...")
    manager.close()

    print("Successful exit.")
    exit()