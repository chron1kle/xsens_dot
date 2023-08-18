import math
import numpy
import nbformat
import torch
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from time import time, sleep, localtime
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Enable auto extra logging if TRUE
enLog = True

# A time period for remaining DOT static before activating it
activateTime = 0.5

# record the functions that supported by each Mode
modeFunc = {'CustomMode1': ['eul', 'facc',
                            'ang_vel'], 'CustomMode4': ['acc', 'eul']}

# Which kinds of data would you like to monitor?
defaultDataOutput = ['eul', 'facc', 'coor', 'vel', 'ori']

# The shreshold of variaty to judge a DOT is static or not
varShreshold = 1.1
flow_threshold = 1
static_hist_threshold = 0.6

# learning rate for the updation of the compensation dictionary
learning_rate = 0.01
hist_len = 10
weight = 0.6

# ......... unused object
class RealtimeWindow:
    global count

    def __init__(self, title, Dots, width=1600, height=800):

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        self.text_var = tk.StringVar()
        self.text_var.set('')

        text_init = tk.StringVar()

        self.text_label = tk.Label(
            self.root, textvariable=self.text_var, font=('Arial', 12))
        self.text_label.grid(row=2, column=0)

        self.btn = tk.Button(self.root, text="Recalibrate",
                             font=("Arial", 12), command=self.velInit)
        self.btn.place(x=550, y=400)

        self.dots = Dots
        '''
        self.canvas = tk.Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.canvas.pack()

        self.text_item = self.canvas.create_text(10, 10, anchor='nw', fill='black', text='')
        '''

    def set_data(self, data):
        self.text_var.set(data)

    def run(self):
        self.root.mainloop()

    def velInit(self):
        for dot in self.dots:
            dot.recalibrate()


class Dot:
    global activateTime, modeFunc, varShreshold, enLog, flow_threshold, static_hist_threshold, hist_len, weight, learning_rate

    def __init__(self, srl, mode) -> None:
        self.previousTime = time()
        self.data_list = []
        self.faccx_list, self.faccy_list, self.faccz_list = [], [], []
        self.ex_list, self.ey_list, self.ez_list = [], [], []
        self.avgAxList, self.avgAyList, self.avgAzList = [], [], []
        self.vel, self.coor, self.cali_vel, self.cali_coor = {}, {}, {}, {}
        self.deltaX = [0, 0, 0]
        self.prev_state, self.end_trail = [], []
        self.mode = mode
        self.srl = srl
        self.activated = False
        self.initialzing = False
        self.hist_len = hist_len
        # Output file path for any data needs to be recorded
        self.logFileName = f"rec_{srl}.txt"
        self.errorLogFileName = f"error_{srl}.log"
        self.compensationDictName = f"cpst_{srl}.dict"
        self.cpstDict = self.load_compensation()

        with open(self.logFileName, '+w'):  # clear the output file and ready for data recording
            pass
        # clear the error output file and ready for data recording
        with open(self.errorLogFileName, '+w'):
            pass
        '''
        while True:
            uin = inumpyut("dataListLen: ",)
            if uin.isdigit() == True:
                self.dataListLen = int(uin)
                break
        while True:
            uin = inumpyut("varListLen: ",)
            if uin.isdigit() == True:
                self.varListLen = int(uin)
                break
        '''
        self.w = torch.tensor([[-0.2236, -0.1852, -0.2851],
                               [0.0722,  0.1348,  0.1330],
                               [0.2414,  0.3205,  0.1915]], requires_grad=True)
        self.b = torch.tensor([[0.5241, 0.2441, 0.7517]], requires_grad=True)

    # Determine if the DOT is currently static
    def is_Static(self) -> bool:
        # return False
        prd = 0
        self.varList = []
        self.varSqrtList = []
        try:
            if 'eul' in modeFunc[self.mode]:
                self.data_list.append(self.ex)
                self.data_list.append(self.ey)
                self.data_list.append(self.ez)
                prd += 3
            '''
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
            '''
        except NameError as e:
            print(f"Exception NameError: {e}")

        if len(self.data_list) < 2 * prd:
            return False
        # calculate the variaty to ensure the DOT remains static
        self.vsum = 0
        s = '\rVarList: '
        for i in range(prd):
            v = numpy.var(self.data_list[-(i + 1):0:-prd])
            self.vsum += round(v, 3)
            s += f'{v:7.3f} '
            self.varList.append(v)
        s += f'| self.Vsum: {self.vsum} '
        '''
        #if len(self.varList) < 2 * prd : return False
        for i in range(prd):
            v2 = numpy.var(self.varList[-(i + 1):0:-prd])
            s += f'{v2:7.3f} '
            self.varSqrtList.append(v2)
            # if v2 > varShreshold: return False
        '''
        #print(s, end='', flush=True)
        if len(self.data_list) > prd * self.hist_len:
            self.data_list = self.data_list[-prd * self.hist_len:]
        if len(self.varList) > prd * self.hist_len:
            self.varList = self.varList[-prd * self.hist_len:]
        try:
            if self.vsum <= varShreshold and abs(self.cali_fax) <= flow_threshold and abs(self.cali_fay) <= flow_threshold and abs(self.cali_faz) <= flow_threshold:
                self.prev_state.append(1)
            else:
                self.prev_state.append(0)
            if len(self.prev_state) > self.hist_len:
                self.prev_state = self.prev_state[1:]
            if self.prev_state.count(1) / len(self.prev_state) >= static_hist_threshold:
                return True
            else:
                return False
        except:
            return True

    def get_Ready(self) -> None:
        if self.initialzing == False:
            self.static_start = time()
            self.initialzing = True

        if self.is_Static() == False:
            self.static_start = time()
            return "Initializing"
        else:
            self.static_end = time()

        # keep the DOT remaining unmoved to activate recording and data processing
        if self.static_end - self.static_start > activateTime:
            self.activated = True
            self.initialzing = False
            self.coor = {'x': 0, 'y': 0, 'z': 0}
            self.vel = {'x': 0, 'y': 0, 'z': 0}
            self.cali_vel = {'x': 0, 'y': 0, 'z': 0}
            self.cali_coor = {'x': 0, 'y': 0, 'z': 0}
            self.static_end, self.static_start = 0, 0
            return "Initialized"
        else:   # self.static_end - self.static_start <= activateTime:
            return "Initializing"

    # The data shown in the command window
    def layout_Data(self) -> str:

        s = f"{'1' if self.state == True else '0'} {self.vsum:6.4f} "
        try:
            s += f"Euler: {self.ex:6.2f} {self.ey:6.2f} {self.ez:6.2f} "
        except:
            pass
        try:
            #s += f"Raw Acc: {self.fax_raw:5.3f} {self.fay_raw:5.3f} {self.faz_raw:5.3f} "
            #s += f"Conv Acc: {self.fax:5.3f} {self.fay:5.3f} {self.faz:5.3f} "
            pass
        except:
            pass
        try:
            #s += f"Calibrated Acc: {self.avgAxList[-1]:6.3f} {self.avgAyList[-1]:6.3f} {self.avgAzList[-1]:6.3f} "
            pass
        except:
            pass
        try:
            #s += f"Vel: {self.vel['x']:6.3f} {self.vel['y']:6.3f} {self.vel['z']:6.3f} "
            pass
        except:
            pass
        try:
            #s += f"Cartisian: {self.ox:6.3f} {self.oy:6.3f} {self.oz:6.3f} "
            s += self.dict_found
            s += f"D_len: {len(self.cpstDict)} "
            pass
        except:
            pass
        try:
            s += f"Coor: {self.coor['x']:5.2f} {self.coor['y']:5.2f} {self.coor['z']:5.2f} "
            # pass
        except:
            pass
        try:
            # s += f"Cali_Coor: {self.cali_coor['x']:5.2f} {self.cali_coor['y']:5.2f} {self.cali_coor['z']:5.2f} "
            pass
        except Exception as e:
            self.write_exception_log(f"Layout cali_coor: {e}")

        if self.state == True:
            if self.prev_state[-1] == 0: 
                self.end_trail.append([self.coor['x'], self.coor['y'], self.coor['z']])
            self.coor = {'x': 0, 'y': 0, 'z': 0}
            self.vel = {'x': 0, 'y': 0, 'z': 0}
            self.cali_vel = {'x': 0, 'y': 0, 'z': 0}
            self.cali_coor = {'x': 0, 'y': 0, 'z': 0}
        return s
        '''
        if self.mode == 'CustomMode1':
            facc = 
            vel = 
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
        '''

    # The data saved to the log
    def save_Log(self) -> None:
        with open(self.logFileName, '+a') as f:
            if enLog == False: pass
            elif self.state == True:
                print(
                    f"{self.ex} {self.ey} {self.ez} {self.deltaX[0]} {self.deltaX[1]} {self.deltaX[2]} {self.fax_raw} {self.fay_raw} {self.faz_raw} {self.avgAxList[-1]} {self.avgAyList[-1]} {self.avgAzList[-1]} 1", file=f)
            else:
                print(
                    f"{self.ex} {self.ey} {self.ez} {self.deltaX[0]} {self.deltaX[1]} {self.deltaX[2]} {self.fax_raw} {self.fay_raw} {self.faz_raw} {self.avgAxList[-1]} {self.avgAyList[-1]} {self.avgAzList[-1]} 0", file=f)
            #print(f"{self.ox} {self.oy} {self.oz} {self.ex} {self.ey} {self.ez} {self.fax} {self.fay} {self.faz}", file=f)
        if self.state == True:
            try:
                self.cpstDict[f"{round(self.ex / 10, 0) * 10},{round(self.ey / 10, 0) * 10},{round(self.ez / 10, 0) * 10}"] = (numpy.array(
                    self.cpstDict[f"{round(self.ex / 10, 0) * 10},{round(self.ey / 10, 0) * 10},{round(self.ez / 10, 0) * 10}"]) * (1 - learning_rate) + numpy.array([self.fax, self.fay, self.faz]) * learning_rate).tolist()
            except:
                self.cpstDict[f"{round(self.ex / 10, 0) * 10},{round(self.ey / 10, 0) * 10},{round(self.ez / 10, 0) * 10}"] = (numpy.array(
                    [self.fax, self.fay, self.faz]) * learning_rate).tolist()
        return

    def calc(self) -> None:
        '''
        try:
            self.ex_list.append(self.ex)
            self.ey_list.append(self.ey)
            self.ez_list.append(self.ez)
            if len(self.ex_list) > hist_len:
                self.ex_list.pop(0)
                self.ey_list.pop(0)
                self.ez_list.pop(0)
            #self.ex, self.ey, self.ez = self.avg(self.ex_list, self.ey_list, self.ez_list, weight)
        except Exception as e:
            self.write_exception_log(f"calc_0: {e}")
        '''
        
        try:
            self.fax_raw, self.fay_raw, self.faz_raw = self.fax, self.fay, self.faz
            self.ox, self.oy, self.oz = self.orientationEuler_to_cartesian([
                                                                           1, 0, 0])
            temp1 = self.orientationEuler_to_cartesian([self.fax, 0, 0])
            temp2 = self.orientationEuler_to_cartesian([0, self.fay, 0])
            temp3 = self.orientationEuler_to_cartesian([0, 0, -self.faz])
            self.fax = temp1[0] + temp2[0] + temp3[0]
            self.fay = temp1[1] + temp2[1] + temp3[1]
            self.faz = temp1[2] + temp2[2] + temp3[2]
        except Exception as e:
            self.write_exception_log(f"calc_1: {e}")
        '''
        try:
            self.deltaX = [self.ox - self.prevOx, self.oy - self.prevOy, self.oz - self.prevOz]
            [self.fax, self.fay, self.faz] = self.project_vector([self.fax, self.fay, self.faz], self.deltaX)
        except Exception as e:
            self.write_exception_log(f"calc_2: {e}")
        '''
        
        '''
        try:
            e = torch.tensor([[self.ex, self.ey, self.ez]])
            fa = torch.tensor([[self.fax, self.fay, self.faz]])
            self.cali_acc = torch.mm(e, self.w) + self.b - fa

            self.dvx = self.cali_acc[0][0].item() * self.freq
            self.dvy = self.cali_acc[0][1].item() * self.freq
            self.dvz = self.cali_acc[0][2].item() * self.freq

            self.faccx_list.append(self.cali_acc[0][0].item())
            self.faccy_list.append(self.cali_acc[0][1].item())
            self.faccz_list.append(self.cali_acc[0][2].item())
            if len(self.faccx_list) > 300:
                self.faccx_list = self.faccx_list[1:]
                self.faccy_list = self.faccy_list[1:]
                self.faccz_list = self.faccz_list[1:]

        except Exception as e:
            print(e)
        '''

        """
        try:
            self.faccx_list.append(self.fax)
            self.faccy_list.append(self.fay)
            self.faccz_list.append(self.faz)
            self.avg_ax, self.avg_ay, self.avg_az = 0, 0, 0
            self.faccx_list = self.faccx_list[-hist_len:]
            self.faccy_list = self.faccy_list[-hist_len:]
            self.faccz_list = self.faccz_list[-hist_len:]
            '''
            for x, y, z in zip(self.faccx_list, self.faccy_list, self.faccz_list):
                self.avg_ax += x
                self.avg_ay += y
                self.avg_az += z
            '''
            self.fax, self.fay, self.faz = self.avg(self.faccx_list, self.faccy_list, self.faccz_list, weight)
            self.avgAxList.append(self.avg_ax)
            self.avgAyList.append(self.avg_ay)
            self.avgAzList.append(self.avg_az)
        except Exception as e:
            self.write_exception_log(f"calc_3: {e}")
        """
        

        try:
            acc_calibration = self.cpstDict[f"{round(self.ex / 10, 0) * 10},{round(self.ey / 10, 0) * 10},{round(self.ez / 10, 0) * 10}"]
            # ------------------ For debugging
            self.dict_found = f"Dict cali: {acc_calibration[0]:6.3f} {acc_calibration[1]:6.3f} {acc_calibration[2]:6.3f} "
        except KeyError:
            acc_calibration = [0, 0, 0]

        try:
            self.dvx = (self.fax - acc_calibration[0]) * self.freq
            self.dvy = (self.fay - acc_calibration[1]) * self.freq
            self.dvz = (self.faz - acc_calibration[2]) * self.freq
            self.cali_fax = self.fax - acc_calibration[0]
            self.cali_dvx = self.cali_fax * self.freq
            self.cali_fay = self.fay - acc_calibration[1]
            self.cali_dvy = self.cali_fay * self.freq
            self.cali_faz = self.faz - acc_calibration[2]
            self.cali_dvz = self.cali_faz * self.freq
        except Exception as e:
            self.write_exception_log(f"calc_4: {e}")

        self.state = self.is_Static()
        
        try:
            self.dpx = (self.vel['x'] * 2 + self.dvx) * self.freq / 2
            self.dpy = (self.vel['y'] * 2 + self.dvy) * self.freq / 2
            self.dpz = (self.vel['z'] * 2 + self.dvz) * self.freq / 2
            self.cali_dpx = (self.cali_vel['x']
                             * 2 + self.cali_dvx) * self.freq / 2
            self.cali_dpy = (self.cali_vel['y']
                             * 2 + self.cali_dvy) * self.freq / 2
            self.cali_dpz = (self.cali_vel['z']
                             * 2 + self.cali_dvz) * self.freq / 2
        except Exception as e:
            self.write_exception_log(f"calc_5: {e}")
        try:
            self.vel['x'] += self.dvx
            self.vel['y'] += self.dvy
            self.vel['z'] += self.dvz
            self.cali_vel['x'] += self.cali_dvx
            self.cali_vel['y'] += self.cali_dvy
            self.cali_vel['z'] += self.cali_dvz
        except Exception as e:
            self.write_exception_log(f"calc_6: {e}")
        
        try:
            self.coor['x'] += self.dpx
            self.coor['y'] += self.dpy
            self.coor['z'] += self.dpz
            self.cali_coor['x'] += self.cali_dpx
            self.cali_coor['y'] += self.cali_dpy
            self.cali_coor['z'] += self.cali_dpz
        except Exception as e:
            self.write_exception_log(f"calc_7: {e}")
        self.prevOx, self.prevOy, self.prevOz = self.ox, self.oy, self.oz
        return

    # main func of this object
    def run(self, dList) -> str:
        for key, value in dList:
            setattr(self, key, value)
        try:
            self.dt = self.t_prev - self.t # delta T between two packets
        except:
            self.dt = 0.005
        self.freq = self.dt / 1000
        self.t_prev = self.t
        
        
        if self.activated == False:
            return self.get_Ready()
        else:
            
            self.calc()
            try:
                self.save_Log()
            except:
                pass
            try:
                s = self.layout_Data()
                self.save_compensation()
                return s
            except Exception as e:
                self.write_exception_log(f"run() error: {e}")
                return ''

    def recalibrate(self) -> None:
        self.__init__(self.mode)
        sleep(1)
        return

    def write_exception_log(self, e) -> None:
        t = localtime()
        with open(self.errorLogFileName, '+a') as f:
            print(f"{t[3]}:{t[4]}:{t[5]} {e}", file=f)
        return

    def save_compensation(self) -> None:
        if round(time(), 0) % 100 == 0:
            with open(self.compensationDictName, 'w') as f:
                json.dump(self.cpstDict, f)
        return

    def load_compensation(self) -> None:
        try:
            with open(self.compensationDictName, 'r') as f:
                d = json.load(f)
            if type(d) == dict:
                return d
            else:
                return {}
        except Exception as e:
            self.write_exception_log(f"load_comp: {e}")
            return {}

    # Convert orientation q into Cartesian vector (x,y,z)
    def orientation_to_cartesian(q):
        x = 2 * (q.x * q.z - q.w * q.y)
        y = 2 * (q.w * q.x + q.y * q.z)
        z = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z
        return x, y, z

    def orientationEuler_to_cartesian(self, vec) -> float:
        # 将角度转换为弧度
        roll = numpy.deg2rad(self.ex)
        pitch = numpy.deg2rad(self.ey)
        yaw = numpy.deg2rad(self.ez)

        # 计算旋转矩阵
        cos_roll = numpy.cos(roll)
        sin_roll = numpy.sin(roll)
        cos_pitch = numpy.cos(pitch)
        sin_pitch = numpy.sin(pitch)
        cos_yaw = numpy.cos(yaw)
        sin_yaw = numpy.sin(yaw)

        R = numpy.array([[cos_yaw*cos_pitch, cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll, cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll],
                         [sin_yaw*cos_pitch, sin_yaw*sin_pitch*sin_roll + cos_yaw *
                             cos_roll, sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll],
                         [-sin_pitch, cos_pitch*sin_roll, cos_pitch*cos_roll]])

        # 计算指向X轴正方向的向量在空间坐标系中的方向向量
        d = numpy.dot(R, numpy.array(vec)).tolist()

        return d[0], d[1], d[2]

    def dot_product(self, a, b) -> float:
        return sum([a[i] * b[i] for i in range(len(a))])

    def vector_length(self, a) -> float:
        return math.sqrt(sum([a[i] ** 2 for i in range(len(a))]))

    # The projection of vec A on vec B
    def project_vector(self, a, b) -> [float]:
        dp = self.dot_product(a, b)
        bl = self.vector_length(b)
        return [(dp / bl ** 2) * b[i] for i in range(len(b))]

    def avg(self, x, y, z, weight) -> float:
        xa, ya, za = 0, 0, 0
        for i, j, k in zip(x, y, z):
            xa += i
            ya += j
            za += k
        return ((xa  / len(x)) * (1 - weight) + x[-1] * weight), ((ya / len(y)) * (1 - weight) + x[-1] * weight), ((za / len(z)) * (1 - weight) + x[-1] * weight)


# ......... unused object
class Plot3d:
    def __init__(self) -> None:
        self.fig = make_subplots(rows=1, cols=1, specs=[
                                 [{'type': 'scatter3d'}]])
        self.colors = ['rgb(255, 0, 0)']
        self.X, self.Y, self.Z = [0], [0], [0]
        self.index_color = 0

    def update_trajectory(self, x, y, z) -> None:

        # 添加轨迹点
        self.X.append(x)
        self.Y.append(y)
        self.Z.append(z)

        # 绘制轨迹
        self.fig.add_trace(
            go.Scatter3d(x=self.X, y=self.Y, z=self.Z, mode='lines',
                         line=dict(color=self.colors[self.index_color])),
            row=1, col=1
        )

        # 更新颜色索引
        # index_color = (index_color + 1) % len(colors)

        # 更新轨迹图
        self.fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))
        self.fig.show()

    def plotTest(self) -> None:
        i = 0
        while True:
            self.update_trajectory(i, i, i)
            i += 1
