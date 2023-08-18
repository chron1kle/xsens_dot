# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import threading, server
from copy import deepcopy
from Sources.config import monitor_ip, plot_port


buffer = []
state = 0
port = plot_port
x_min, x_max, y_min, y_max, z_min, z_max = -1, 1, -1, 1, -1, 1
clear_threshold = 5

x, y, z = [0], [0], [0]
def update() -> None:
    global x, y, z, state, x_min, x_max, y_min, y_max, z_min, z_max, clear_threshold
    
    try:
        while True:
            #s = srvr.receiving()
            print(f'\r{len(buffer)}', end='', flush=True)
            new_data = buffer.pop(0)#s.split(' ')
            discard1 = buffer.pop(0)
            discard2 = buffer.pop(0)
            discard3 = buffer.pop(0)
            #print(' ',state, flush=True)

            if new_data == ['0', '0', '0']:
                state += 1
                #print(f'\t{state}')
                
            elif state > clear_threshold:
                state = -1
                #print(f'\t{state}')
                x, y, z = [0], [0], [0]
                x_min, x_max, y_min, y_max, z_min, z_max = -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
                break
            else:
                a, b, c = round(float(new_data[0]), 2), round(-float(new_data[1]), 2), round(float(new_data[2]), 2)
                if a > x_max:
                    x_max = deepcopy(a)
                elif a < x_min:
                    x_min = deepcopy(a)
                if b > y_max:
                    y_max = deepcopy(b)
                elif b < y_min:
                    y_min = deepcopy(b)
                if c > z_max:
                    z_max = deepcopy(c)
                elif c < z_min:
                    z_min = deepcopy(c)
                x.append(a)
                y.append(b)
                z.append(c)
                #print(f'\t{state}', flush=True)
                state = 0
                break
    except Exception as e:
        #print(f'not appended: {e}', end='', flush=True)
        pass
    if state == -1: 
        return True
    return False

def show_ori() -> None:
    global x, y, z
    try:
        new_data = buffer.pop(0)#s.split(' ')
        discard1 = buffer.pop(0)
        discard2 = buffer.pop(0)
        discard3 = buffer.pop(0)
        a, b, c = float(new_data[0]), float(new_data[1]), float(new_data[2])
        mod = (a**2 + b**2 + c**2) ** 0.5
        x = [0.0, round((a / mod), 2)]
        y = [0.0, round((b / mod), 2)]
        z = [0.0, round((c / mod), 2)]
    except Exception as e:
        pass
    return

srvr = server.srvr_sckt((monitor_ip, port))
def tst():
    global buffer, srvr
    while True:
        s = srvr.receiving()
        buffer.append(s.split(' '))
    return
t = threading.Thread(target=tst)
t.start()

frame_counter = 0
def animate_func(num):
    global x, y, z, frame_counter, x_min, x_max, y_min, y_max, z_min, z_max

    if port in [12571, 12575] and update():
        print('new trail')
        plt.cla()

    if port in [12570, 12572, 12574]:
        plt.cla()
        show_ori()
    
    ax.clear()  # Clears the figure to update the line, point,   
                # title, and axes   
    ax.plot3D(x, y, z, c='blue')    # Updating Point Location
    ax.scatter(x[-1], y[-1], z[-1],
            c='red', marker='o')    # Adding Constant Origin
    ax.plot3D(x[0], y[0], z[0],     
            c='black', marker='o')    # Setting Axes Limits
    ax.set_xlim3d(x_min, x_max)#([min(x), max(x)])
    ax.set_ylim3d(y_min, y_max)#([min(y), max(y)])
    ax.set_zlim3d(z_min, z_max)#([min(z), max(z)])

    # Adding Figure Labels
    ax.set_title('Trajectory \nPackets = ' + str(frame_counter))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    frame_counter += 1

# Plotting the Animation
fig = plt.figure()
ax = plt.axes(projection='3d')
line_ani = animation.FuncAnimation(fig, animate_func, interval=1)
plt.show()