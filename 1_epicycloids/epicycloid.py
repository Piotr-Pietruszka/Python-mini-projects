import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fig = plt.figure()
# ax = plt.axes(xlim = (-10, 10), ylim = (-10,10))
# line, = ax.plot([], [])

print("Give R: ")
R = float(input())
print("Give r: ")
r = float(input())
# R = 2
# r = 1


fig = plt.figure()
ax = plt.axes(xlim = (-1.5*(R+r), 1.5*(R+r)), ylim = (-1.5*(R+r),1.5*(R+r)))
line, = ax.plot([], [])
line2, = ax.plot([], [])

circ_big = plt.Circle((0,0), R, facecolor='None', edgecolor='r')
circ_small = plt.Circle((R+r,0), r, facecolor='None', edgecolor='g')

x_data, y_data = [], []

def init():
    line.set_data([], [])
    line2.set_data([], [])
    ax.add_patch(circ_small)
    return line, line2


def animate(i):
    t = 0.1*i
    #R = 6
    #r = 2
    x = (R+r)*np.cos(t) - r*np.cos((R+r)/r*t)
    y = (R+r)*np.sin(t) - r*np.sin((R+r)/r *t)

    x_b = (R+r) * np.cos(t)
    y_b = (R+r) * np.sin(t)


    x_data.append(x)
    y_data.append(y)

    line2.set_data([x_b, x], [y_b, y])
    line.set_data(x_data, y_data)
    circ_small.center = (x_b,y_b)
    return line, circ_small, line2

plt.gca().add_patch(circ_big)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=50, blit=False)
def on_click(event):
    #global anim_running
    anim.event_source.stop()

def on_release(event):
    anim.event_source.start()


fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()