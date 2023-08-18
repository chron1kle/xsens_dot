

# This is an introduction of a sample development package for Xsens DOT, based on a demo by Xsens DOT SDK for python.

By Hu Yixin, Saunders  
Exclusively based on `Python 3.10`


***

## 1. Installation of SDK and demo program

The original sample program and SDK could be downloaded from [here](https://www.movella.com/support/software-documentation).

The sample program is included in the SDK file.

### SDK directory structure
```
└───DOT PC SDK 2022.2
    ├───Documentation          // This is the documentation for the SDK library
    │   └───XDPC SDK
    │       ├───xscontroller
    │       │   └───doc
    │       │       └───html
    │       │           └───search
    │       ├───xsensdot_pc_sdk
    │       │   └───doc
    │       │       └───html
    │       │           └───search
    │       └───xstypes
    │           └───doc
    │               └───html
    │                   └───search
    └───SDK Files
        ├───Debugging
        │   └───MSVS
        ├───Examples          // Below are the demo provided by Xsens in different programming languages.
        │   ├───cpp
        │   ├───csharp
        │   │   ├───extralibs
        │   │   │   └───x64
        │   │   ├───Properties
        │   │   └───wrap_csharp64
        │   ├───java
        │   │   └───src
        │   │       └───com
        │   │           └───xsens
        │   │               └───xsensdot_pc_sdk
        │   └───python       // The original README.MD by Xsens is under this directory.
        ├───Python
        │   └───x64          // This is where the .whl files locate. Run 'pip install xxx.whl' to install SDK library.
        ├───wrap_csharp64
        └───x64
            ├───include
            │   ├───xscommon
            │   ├───xscontroller
            │   ├───xsensdot_pc_sdk
            │   └───xstypes
            └───lib
```


***
## 2. Guidance on the DOT development program package

This program is modified and improved on the basis of the demo programs by Xsens DOT SDK.

Up to 3 hosts are included when running the program package, including `client`, `server`, and `monitor`. The 3 could be a same host physically, but it's recommended to prepare at least 2 hosts, and ensure they're under same network.


### The basic logic structure of the development program
1. Connect DOT to `client` through BLE. The `client` host should support BLE service.  
2. `client` initializes the DOT, and fetches the raw data from DOT.  
3. `client` passes the raw data to `server` through socket, for further data processing.
4. `server` processes the data, and passes the data to `monitor` for visualization.
5. `monitor` receives the processed data and displays them.

### Details on specific operation when running the program
1. There are 3 scripts separately for the 3 hosts to execute directly:  
    - `launcher.py` --> `client`
    - `server.py`   --> `server`
    - `3dplot.py`   --> `monitor`
2. The order of running scripts is important. Please run `server.py` on host `server` first, then `launcher.py` on `client`, because there's a handshake machanism between `client` and `server` before passing the raw data. `3dplot.py` could be executed after the connection between `server` and `client` is established.
3. Before running the scripts mentioned above, you should first set the configuration in `./Source/config.py`, including the IP addresses and the ports of the three hosts.
4. After the 3 scripts are finely executed:
    - The command window of `clients` will keep refreshing to show the total number of packets sent, from which you could calculate the sample rate by dividing this number by running time.
    - The command window of `server` will keep refreshing to show the processed data of DOT. You could manually select the data you want to monitor by modifying the code in `proc.py`(Details will be demonstrated later). At the same time, a chart would appear to show the changing of certain data in real-time. Likely, the data could be selected by modifying the code in `server.py`.
    - A 3D plot would appear after `3dplot.py` is executed, showing the real-time trail of the DOT. Likely, this could be changed to the *orientation* or something else manually in `server.py`.


***
## 3. Specific details on each scripts

This part contains details of each scripts within this development package, for any further development.

1. `launcher.py`
    - Global variable `totalDots`: determining the max num of DOTs to be connected. By default, the program will wait for all DOTs being connected before procceeding.
    - At the front is a dict called `MODES`, contains all the modes supported by the DOT. When initializing the DOT, developer should choose a certain mode for the DOT to let it pass back certain data. Further details could be found in the SDK document.
    - Global vairable `defaulteMode`: the mode taken as default, for simplification for the development process.
    * After the connection between `server` and `client` is established, developer may come up with a situation that they want to make some modifications in scripts related with `server` or `monitor` but not with `client`, which implies he would not like to terminate the whole process of `launcher.py`. Given this circumstances, he could directly press `Ctrl + C` to pause `client`, until he finishes his modifications and the program could resume. But remember: start the  `server.py` on `server` first before resume `launcher.py`!
    * The DOT could pass back data: Euler orientation, Acceleration, Free acceleration, angular velocity, magnatic field.
    * **Please carefully refer to the prompt in the cmd window when running the script!**
2. `proc.py`
    - This script is for the data processing on `server`, related with `server.py`.
    - To process the raw data, the program perform a transition from *Euler Orientation* to *Cartisian Vector* to represent the orientation of DOT. This data is basicly accurate.
    - However, there're really much noises in the data *Acceleration* and  *Free acceleration*. First, due to the gyroscope's inherent error, the three-axis acceleration continued to fluctuate even in a stationary state. Second, the gyroscope had a fixed error, meaning that in a stationary state, the three-axis acceleration did not oscillate around zero but around a non-zero value. Third, this fixed error changed with the orientation, meaning that the resultant force of the three-axis acceleration in a stationary state varied with different orientations. Fourth, the error in the data returned by the sensor differed in different motion states and was unrelated to the orientation. More intense movements were more likely to cause significant errors, and larger errors persisted for a longer time in the time series. After a vigorous movement stopped, more time was needed to return to a smaller normal error. 
    - To filter the noises of the data, the program perform a recognizer to judge if DOT is actually static or not. If yes, the velocity and coordinate of the DOT will be forced to be zero. The judgement is according to the variance of DOT's history orientation and acceleration. Global variable `varShreshold` and `flow_threshold` are used to controll the threshold of orientation and acceleration separately. Variable `hist_len` determines the length of the list of historical data for calculation.
    - To further calibrate the data, the program has a dict under `./Sources/cpst_x.dict`. It records the acceleration in different orientation when static. That is, to substract the coresponding value in the dict from the real acceleration data, we could get a more precise acc data. **However, this approach is not working well actually. The program was set to refresh the value in the dict whenever DOT is static by a certain `learning_rate`, but this function seems not working. This may be some error in the code.**
3. `server.py`
    - Developer could change the data to be shown in the chart and the 3D plot in the script.
4. `config.py`
    - Variable `plot_port` chooses the data that will be finally shown in the 3D plot(please refer to the code in `server.py`). This step could let developer doesn't have to terminate `server.py` when changing the data to be shown in the 3D plot.
5. `CallBackHandler.py` and `user_settings.py`
    - please do not perform any modifictaions to these two scripts if unnecessary.
6. `dataProcess.py`, `learning.py` and `learning_a.py` are not in use.