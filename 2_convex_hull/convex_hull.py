from matplotlib import pyplot as plt
import numpy as np


def left_side(x_t, y_t, p1, o1, o2):
    left_side_no = (x_t[o2]-x_t[o1])*(y_t[p1]-y_t[o1]) - (x_t[p1]-x_t[o1])*(y_t[o2]-y_t[o1])
    if left_side_no >= 0:
        return True
    else:
        return False



if __name__ == "__main__":

    xmin, xmax = 0, 100
    ymin, ymax = 0, 100
    n = 20

    x_t = np.random.rand(n)*(xmax-xmin) + xmin
    y_t = np.random.rand(n) * (ymax - ymin) + ymin




    hull_list = []
    hull_point = np.argmax(x_t)
    print(f"P0: {hull_point}")
    i = 0
    while True:
        hull_list.append(hull_point)
        next_point = 0  # nowy punktu - tego sprawdzamy i jesli jest jakis na lewo to zmieniamy
        for j in range(0, len(x_t)):
            #print(f"\nfor: {j}")
            if j != hull_list[i] and left_side(x_t, y_t, j, hull_list[i], next_point):
                next_point = j
                #print(f"if: {j}")
        hull_point = next_point

        i = i+1  # by miec dostep do punktu poprzedniego (hull_list[i])
        if next_point == hull_list[0]:
            hull_list.append(next_point)
            break


    print(hull_list)
    x_hull = np.array(x_t)[hull_list]
    y_hull = np.array(y_t)[hull_list]

    plt.scatter(x_t, y_t)
    plt.scatter(x_hull, y_hull)
    plt.plot(x_hull, y_hull)
    plt.show()
    print(x_t)


