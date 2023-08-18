import math, numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from time import time, sleep

# Gravity acceleration
g = 9.81

# set the frequency of the data for processing to 100Hz
freq = 0.01 

# Extra output file path for any processed data (if needed), except for log.csv as the raw data log
logFileName = ".\\rec.txt" 

# Enable auto extra logging if TRUE
enLog = True

# A time period for remaining DOT static before activating it
activateTime = 5

# record the functions that supported by each Mode
modeFunc = {'CustomMode1':['eul', 'facc', 'ang_vel'], 'CustomMode4': ['acc', 'eul']}

# Which kinds of data would you like to monitor? 
defaultDataOutput = ['eul', 'facc', 'coor', 'vel', 'ori']

# The shreshold of variaty to judge a DOT is static or not
varShreshold = 0.001

#
dataListLen = 10
varListLen = 10

class RealtimeWindow:
    global count
    
    def __init__(self, title, Dots, width=1600, height=800):

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        self.text_var = tk.StringVar()
        self.text_var.set('')

        text_init = tk.StringVar()


        self.text_label = tk.Label(self.root, textvariable=self.text_var, font=('Arial', 12))
        self.text_label.grid(row=2, column=0)

        self.btn = tk.Button(self.root, text = "Recalibrate", font=("Arial", 12), command=self.velInit)
        self.btn.place(x=550, y=400)

        self.dots = Dots
        '''
        self.canvas = tk.Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.canvas.pack()

        self.text_item = self.canvas.create_text(10, 10, anchor='nw', fill='black', text='')
        '''
    def set_data(self, data):
        """
        参数：
        data - 要输出的数据（字符串格式）
        self.canvas.itemconfigure(self.text_item, text=data)
        """
        self.text_var.set(data)

    def run(self):
        self.root.mainloop()

    def velInit(self):
        for dot in self.dots:
            dot.recalibrate()

class Dot:
    global g, freq, logFileName, activateTime, modeFunc, varShreshold, enLog, varListLen, dataListLen

    def __init__(self, mode) -> None:
        self.previousTime = time()
        self.data_list = []
        self.mode = mode
        self.activated = False
        self.initialzing = False
        self.varList = []
        with open(logFileName, '+w'):  # clear the output file and ready for data recording
            pass
        while True:
            uin = input("dataListLen: ",)
            if uin.isdigit() == True:
                self.dataListLen = int(uin)
                break
        while True:
            uin = input("varListLen: ",)
            if uin.isdigit() == True:
                self.varListLen = int(uin)
                break

    # Determine if the DOT is currently static
    def is_Static(self) -> bool:
        return True
        prd = 0
        self.varSqrtList = []
        try:
            if 'eul' in modeFunc[self.mode]:
                self.data_list.append(self.ex)
                self.data_list.append(self.ey)
                self.data_list.append(self.ez)
                prd += 3
            if 'facc' in modeFunc[self.mode]:
                self.data_list.append(self.fax)
                self.data_list.append(self.fay)
                self.data_list.append(self.faz)
                prd += 3
            if 'acc' in modeFunc[self.mode]:
                self.data_list.append(self.ax)
                self.data_list.append(self.ay)
                self.data_list.append(self.az)
                prd += 3
        except NameError as e:
            print(f"Exception NameError: {e}")
        
        if len(self.data_list) < 2 * prd : return False
        # calculate the variaty to ensure the DOT remains static
        for i in range(prd):
            v = numpy.var(self.data_list[-(i + 1):0:-prd])
            self.varList.append(v)
        if len(self.varList) < 2 * prd : return False
        for i in range(prd):
            v2 = numpy.var(self.varList[-(i + 1):0:-prd])
            self.varSqrtList.append(v2)
            if v2 > varShreshold: return False

        if len(self.data_list) > prd * self.dataListLen:
            self.data_list = self.data_list[-prd * self.dataListLen:]
        if len(self.varList) > prd * self.varListLen:
            self.varList = self.varList[-prd * self.varListLen:]
        return True
    
    def get_Ready(self) -> None:
        if self.initialzing == False:
            self.static_start = time()
            self.initialzing = True

        if self.is_Static() == False:
            vs = ''
            for v2 in self.varSqrtList:
                vs += f"{v2:6.3f} "
            print(f"\rVar Sqrt {len(self.varSqrtList)}: {vs} ", end = '', flush=True)
            self.static_start = time()
            return "Initializing"
        else:
            self.static_end = time()

        if self.static_end - self.static_start > activateTime:  # keep the DOT remaining unmoved to activate recording and data processing
            self.activated = True
            self.initialzing = False
            self.coor = {'x':0, 'y':0, 'z':0}
            self.vel = {'x':0, 'y':0, 'z':0}
            return "Initialized"
        else:   # self.static_end - self.static_start <= activateTime:
            return "Initializing"
    
    def layout_Data(self) -> str:
        s = ""
        if self.mode == 'CustomMode1':
            facc = f"Free Acc: {self.fax:7.2f} {self.fay:7.2f} {self.faz:7.2f}\t"
            vel = f"Velocity: {self.vel['x']:7.2f} {self.vel['y']:7.2f} {self.vel['z']:7.2f}\t"
            # ori = f"Orientation (Cartisian): {self.ox:7.2f} {self.oy:7.2f} {self.oz:7.2f}\t"
            coor = f"Coordinate: {self.coor['x']:7.2f} {self.coor['y']:7.2f} {self.coor['z']:7.2f}\n"
            eul = f"Euler: {self.ex:9.6f} {self.ey:9.6f} {self.ez:9.6f}"
            deul = f"delta euler: {abs(self.ex - self.prevEx):9.6f} {abs(self.ey - self.prevEy):9.6f} {abs(self.ez - self.prevEz):9.6f}"
            s += facc + vel + deul + eul
            return s
        elif self.mode == 'CustomMode4':
            acc = f"Acc: {self.ax:7.2f} {self.ay:7.2f} {self.az:7.2f}\t"
            coor = f"Coordinate: {self.coor['x']:7.2f} {self.coor['y']:7.2f} {self.coor['z']:7.2f}\n"
            eul = f"Euler: {self.ex:9.6f} {self.ey:9.6f} {self.ez:9.6f}"
            s += acc + eul + coor
            return s
    
    def save_Log(self) -> None:
        with open(logFileName, '+a') as f:
            print(f"{abs(self.ex - self.prevEx)} {abs(self.ey - self.prevEy)} {abs(self.ez - self.prevEz)}", file = f)
            #print(f"{self.ox} {self.oy} {self.oz} {self.ex} {self.ey} {self.ez} {self.fax} {self.fay} {self.faz}", file=f)
        return
    
    def calc(self) -> None:
        
        if self.is_Static() == True and False:
            self.fax, self.fay, self.faz = 0, 0, 0
        self.ox, self.oy, self.oz = self.orientationEuler_to_cartesian()

        if self.mode == 'CustomMode1':
            self.dvx = self.fax * freq
            self.dvy = self.fay * freq
            self.dvz = self.faz * freq

            self.dpx = (2 * self.vel['x'] + self.dvx) * freq
            self.dpy = (2 * self.vel['y'] + self.dvy) * freq
            self.dpz = (2 * self.vel['z'] + self.dvz) * freq

            self.vel['x'] += self.dvx
            self.vel['y'] += self.dvy
            self.vel['z'] += self.dvz

            self.coor['x'] += self.dpx
            self.coor['y'] += self.dpy
            self.coor['z'] += self.dpz

            self.prevEx, self.prevEy, self.prevEz = self.ex, self.ey, self.ez
            return
        elif self.mode == 'CustomMode4':
            return
    
    def run(self, **kwargs) -> str:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.prevEx, self.prevEy, self.prevEz = self.ex, self.ey, self.ez
        if self.activated == False:
            return self.get_Ready()
        else:
            self.calc()
            if enLog == True: self.save_Log()
            s = self.layout_Data()
            return s
    
    def recalibrate(self) -> None:
        self.__init__(self.mode)
        sleep(1)
        return

    # Convert orientation q into Cartesian vector (x,y,z)
    def orientation_to_cartesian(q):
        x = 2 * (q.x * q.z - q.w * q.y)
        y = 2 * (q.w * q.x + q.y * q.z)
        z = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z
        return x, y, z

    def orientationEuler_to_cartesian(self) -> float:
        # degree to rad
        roll, pitch, yaw = self.ex, self.ey, self.ez
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)
        roll = math.radians(roll)

        x = math.cos(yaw) * math.cos(pitch)
        y = math.sin(yaw) * math.cos(pitch)
        z = math.sin(pitch)

        return x, y, z



'''
def accRecord(fax, fay, faz, ex, ey, ez, ox, oy, oz):
    with open(".\\rec.txt", '+a') as f:
        print(f"{ox} {oy} {oz} {ex} {ey} {ez} {fax} {fay} {faz}", file=f)
    return

def acceleration_in_object_frame(phi, theta, ax, ay, az, g):
    """
    给定欧拉角和重力加速度大小，计算在物体自身坐标系下各个方向的加速度大小。

    参数：
    phi - 物体绕Z轴旋转的角度（弧度）
    theta - 物体绕Y轴旋转的角度（弧度）
    g - 重力加速度大小（m/s^2）

    返回值：
    返回一个三元组，表示在物体自身坐标系下各个方向的加速度大小（x, y, z）。
    """
    # 计算旋转矩阵
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    R = np.dot(R_y, R_z)

    # 计算重力加速度向量（在物体坐标系下）
    g_object = np.array([0, 0, g])
    
    # 计算在物体自身坐标系下各个方向的加速度大小
    a_object = np.dot(R, g_object)
    ag = np.array([-ax, ay, -az])
    act_a = ag - a_object
    return act_a

# Calculate current velocity
def calc(fax, fay, faz):
    global vel
    acc_shreshold = 0
    vel_shreshold = 0
    dvx, dvy, dvz = 0, 0, 0
    if abs(fax) > acc_shreshold: dvx = fax * freq
    if abs(fay) > acc_shreshold: dvy = fay * freq
    if abs(faz) > acc_shreshold: dvz = faz * freq

    dpx = (2 * vel['x'] + dvx) * freq
    dpy = (2 * vel['y'] + dvy) * freq
    dpz = (2 * vel['z'] + dvz) * freq

    vel['x'] += dvx
    vel['y'] += dvy
    vel['z'] += dvz
    if abs(vel['x']) < vel_shreshold: vel['x'] = 0
    if abs(vel['y']) < vel_shreshold: vel['y'] = 0
    if abs(vel['z']) < vel_shreshold: vel['z'] = 0
    coor['x'] += dpx
    coor['y'] += dpy
    coor['z'] += dpz
    return 

# determine whether DOTS remain static
def static(fax, fay, faz, ex, ey, ez, ox, oy, oz):
    global data_list, previousTime, currentTime, count
    if count > 100:  #  five seconds
        return True
    #accRecord(fax, fay, faz, ex, ey, ez, ox, oy, oz)
    for i in [ox, oy, oz, ex, ey, ez, fax, fay, faz]:
        data_list.append(i)
        if len(data_list) > 1000:
            data_list = data_list[-1001:-1]
    # Already have: Orientation, Euler, Free Acceleration
    ox_var = numpy.var(data_list[-9:-900:-9])
    oy_var = numpy.var(data_list[-8:-908:-9])
    oz_var = numpy.var(data_list[-7:-907:-9])
    #ex_var = numpy.var(data_list[-6:-906:-9])
    #ey_var = numpy.var(data_list[-5:-905:-9])
    #ez_var = numpy.var(data_list[-4:-904:-9])
    fax_var = numpy.var(data_list[-3:-903:-9])
    fay_var = numpy.var(data_list[-2:-902:-9])
    faz_var = numpy.var(data_list[-1:-901:-9])
    with open(".\\rec.txt", '+a') as f:
        print(f"{ox_var} {oy_var} {oz_var} {fax_var} {fay_var} {faz_var}", file = f)
    if ox_var < 0.001 and oy_var < 0.001 and oz_var < 0.001 and fax_var < 0.001 and fay_var < 0.001 and faz_var < 0.001:
        count += 1
    else:
        count = 0
    # {
    # s += f"Ori_x Var: {:7.3f} Ori_y Var: {:7.3f} Ori_z Var: {:7.3f} \n"
    # s += f"Eul_x var: {:7.3f} Eul_y var: {:7.3f} Eul_z var: {:7.3f} \n"
    # s += f"Facc_x var: {:7.3f} Facc_y var: {:7.3f} Facc_z var: {:7.3f} \n"
    # currentTime = time()
    # s += f"Time Lapse: {currentTime - previousTime} seconds.\n"
    # previousTime = currentTime
    # }
    
    return False

# Choose the data you want to be showed on terminal
def selectData():
    dataMonitor = {"ori": 0, "eul": 0, "acc": 0, "dvel": 0, "vel": 0, "facc": 0}
    s = ""
    print("\n------------------------------------")
    print("Please select the data you want to be showed on terminal. Please ipnut the codenames in \033[34mblue\033[0m and separat them by a space.")
    print("\033[34mori\033[0m: Orientation (Cartisian)")
    print("\033[34meul\033[0m: Orientation (Euler)")
    print("\033[34macc\033[0m: Calibrated Acceleration")
    print("\033[34mdvel\033[0m: Delta Velocity")
    print("\033[34mvel\033[0m: Velocity")
    print("\033[34mfacc\033[0m: Free Acceleration")
    print("------------------------------------\n")
    while True:
        uin = input()
        uin = uin.split(' ')
        retry = 0
        for i in uin:
            if i in dataMonitor.keys():
                dataMonitor[i] = 1
            else:
                print(f"Input '{i}' invalid. Please retry.")
                retry = 1
                break
        if retry == 0: break
    return dataMonitor

def printData(dataMon, **kwargs):
    s = ""
    for key, value in kwargs.items():
        if key in dataMon: s += value
    return s

def processing(mode, **kwargs):
    s = ''
    
    if mode == 'CustomMode1':
        for key, value in kwargs.items():
            if key == 'ex': ex = value
            elif key == 'ey': ey = value
            elif key == 'ez': ez = value
            elif key == 'fax': fax = value
            elif key == 'fay': fay = value
            elif key == 'faz': faz = value
        ox, oy, oz = orientationEuler_to_cartesian(ex, ey, ez)
        if static(fax, fay, faz, ex, ey, ez, ox, oy, oz) == True:
            calc(fax, fay, faz)
            s += f"Free Acc: {fax:7.2f} {fay:7.2f} {faz:7.2f}\t"
            s += f"Vel: {vel['x']:7.2f} {vel['y']:7.2f} {vel['z']:7.2f}\t"
            s += f"Orientation: {ox:7.2f} {oy:7.2f} {oz:7.2f}\t"
            s += f"Coordinate: {coor['x']:7.2f} {coor['y']:7.2f} {coor['z']:7.2f}\n"
        if False:
            ori = f"Ori: {ox:7.2f}, {oy:7.2f}, {oz:7.2f}|\t"
            eul = f"Euler: {ex:7.2f}, {ey:7.2f}, {ez:7.2f}|\t"
            facc = f"facc: {fax:7.2f}, {fay:7.2f}, {faz:7.2f}|\t"
            s += ori + eul + facc + '\n'
    
    elif mode == 'CustomMode4':
        for key, value in kwargs.items():
            if key == 'ex': ex = value
            elif key == 'ey': ey = value
            elif key == 'ez': ez = value
            elif key == 'vx': vx = value
            elif key == 'vy': vy = value
            elif key == 'vz': vz = value
            elif key == 'ax': ax = value
            elif key == 'ay': ay = value
            elif key == 'az': az = value
        act_a = acceleration_in_object_frame(ez, ey, ax, ay, az, -g)
        acc = f"Acc: {ax:7.2f}, {ay:7.2f}, {az:7.2f}, {act_a[0]:7.2f}, {act_a[1]:7.2f}, {act_a[2]:7.2f}| "
        dvel= f"dv: {vx:7.2f}, {vy:7.2f}, {vz:7.2f}|"
    
    return s

'''