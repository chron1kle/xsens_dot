import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
import threading, time

class cht:
    def __init__(self, plotNum) -> None:
        self.root = tk.Tk()
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.fig = plt.figure()
        self.fig.set_size_inches(16, 9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.plots = []
        self.lines = []
        self.xdata = [] 
        self.ydata = []
        for i in range(plotNum):
            self.add_plot()

    def add_plot(self) -> None:
        self.plots.append(self.fig.add_subplot(3, 3, len(self.plots) + 1))
        self.plots[-1].set_xlim([0, 100])
        self.plots[-1].set_ylim([0, 100])
        tmp_line, = self.plots[-1].plot([], [], 'b-')
        self.lines.append(tmp_line)
        
        self.xdata.append([0])
        self.ydata.append([0])

    def exe(self) -> None:
        ani = FuncAnimation(self.fig, self.update, frames=60, interval=10)
        tk.mainloop()
        
    def update(self, frame) -> FuncAnimation:
        for i in range(len(self.plots)):
            y_min = min(self.ydata[i])
            y_max = max(self.ydata[i])
            if abs(y_min) >= abs(y_max):
                self.plots[i].set_ylim([y_min, -y_min])
            else:
                self.plots[i].set_ylim([-y_max, y_max])
            self.plots[i].set_xlim([self.xdata[i][0], self.xdata[i][-1]])
            self.lines[i].set_data(self.xdata[i], self.ydata[i])
        return tuple(self.lines)

    def run(self, y, i) -> None:
        self.ydata[i].append(y)
        self.xdata[i].append(time.time())
        if len(self.ydata[i]) > 300:
            self.xdata[i] = self.xdata[i][1:]
            self.ydata[i] = self.ydata[i][1:]

        
'''
t = cht(1)
def re():
    for i in range(20):
        t.run(i, 0)
        print(i)
        time.sleep(0.5)
q = threading.Thread(target=re)
q.start()
t.exe()
'''