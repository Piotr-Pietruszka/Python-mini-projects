import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import cos, sin, pi, e



r = 0.25
N = 2000
t_beg = -2
t_end = 3
more_parameters = False

print("Give x(t): ")
x_func_s = input()
print("Give y(t): ")
y_func_s = input()

if more_parameters:
    print("Give first value of parameter t: ")
    t_beg = float(input())
    print("Give last value of parameter t: ")
    t_end = float(input())
    print("Give number of steps: ")
    N = int(input())
    print("Give radius of small circle: ")
    r = float(input())

ratio = (t_end - t_beg)/N
x_func = lambda t: eval(x_func_s)
y_func = lambda t: eval(y_func_s)


t_array = np.linspace(t_beg, t_end, N)
x_array = x_func(t_array)
y_array = y_func(t_array)

fig = plt.figure()
xy_min = np.amin([np.amin(x_array) - 2*r, np.amin(y_array) - 2*r])
xy_max = np.amax([ np.amax(x_array) + 2*r, np.amax(y_array) +2*r])
# ax = plt.axes(xlim = (xy_min, xy_max),
#                ylim = (xy_min, xy_max))

ax = plt.axes(xlim = (np.amin(x_array) - 2*r, np.amax(x_array) + 2*r),
              ylim = (np.amin(y_array) - 2*r, np.amax(y_array) +2*r))
#ax = plt.axes()
line, = ax.plot([], [])
line2, = ax.plot([], [])

circ_small = plt.Circle((r,0), r, facecolor='None', edgecolor='g')


x_data, y_data = [], []

def init():
    line.set_data([], [])
    line2.set_data([], [])
    ax.add_patch(circ_small)
    return line, line2


length = np.pi

def animate(i):

    i = i + 1
    i_m = i-1
    i_p = i+1

    t =  ratio * i + t_beg
    tm = ratio * i_m + t_beg
    tp = ratio * i_p + t_beg


    # _c - krzywa po ktorej toczy sie kolo
    # ----------------------------
    x_c = x_func(t)
    y_c = y_func(t)

    x_cm = x_func(tm)
    y_cm = y_func(tm)

    x_cp = x_func(tp)
    y_cp = y_func(tp)
    #----------------------------

    # [x, y] prostopadly do [-y, x]
    i_x = (y_cp-y_cm)/ np.sqrt((y_cp-y_cm)**2 + (x_cp - x_cm)**2)
    i_y = -(x_cp - x_cm)/ np.sqrt((y_cp-y_cm)**2 + (x_cp - x_cm)**2)

    # _s - wspolrzedne srodka kola
    # ----------------------------
    x_s = x_c + i_x*r
    y_s = y_c + i_y*r
    # ----------------------------

    #x, y - wspolrzedne krzywej
    # ----------------------------
    global length
    length += (np.sqrt((x_c-x_cm)**2 + (y_c-y_cm)**2))/r
    length = length % (2*np.pi)

    x = x_s + np.cos(length)*r
    y = y_s + np.sin(length)*r

    # ----------------------------


    x_data.append(x)
    y_data.append(y)


    line.set_data(x_data, y_data)
    line2.set_data([x_s, x], [y_s, y])
    circ_small.center = (x_s,y_s)
    return line, circ_small, line2

plt.plot(x_func(t_array), y_func(t_array))

print(20000/N)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=N, interval=8000/N, repeat =False)
def on_click(event):
    #global anim_running
    anim.event_source.stop()

def on_release(event):
    anim.event_source.start()


fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()