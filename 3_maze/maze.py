import numpy as np
from numpy.random import randint as rand
import matplotlib.pyplot as plt
import random


stack = []
visited = []

node_count = 8
r = 20
size = 2*r

class Cell:
    def __init__(self, x_center, y_center):
        self.x_center = x_center
        self.y_center = y_center
        self.opened_walls = [False, False, False, False, False, False, False, False]
        self.points = []
        for i in range(node_count):
            self.points.append([int(r + self.x_center*(size-3) + r * (np.cos(np.pi/8+(i * 2 * np.pi)/ node_count))),
                                int(r + self.y_center*(size-3) + r * (np.sin(np.pi/8+(i * 2 * np.pi)/ node_count)))])

    def open_wall(self, wall_nr):
        self.points[wall_nr][0] = - self.points[wall_nr][0]
        self.points[wall_nr][1] = - self.points[wall_nr][1]
        if self.opened_walls[wall_nr] == False:
            self.opened_walls[wall_nr] = True
        else:
            self.opened_walls = False

def create_maze(my_cells):
    x = 0
    y = 0
    stack.append((x, y))  # lista komórek, z ktorych mozna przejsc do nieodwiedzonych komorek
    visited.append((x, y))  # odwiedzone komorki
    while len(stack) > 0:
        # Stworzenie listy nieodwiedzonych sąsiadów
        neighbours = []
        if (x + 1, y) not in visited and x + 1 < maze_size:
            neighbours.append("right")
        if (x + 1, y + 1) not in visited and x + 1 < maze_size and y + 1 < maze_size:
            neighbours.append("right up")
        if (x + 1, y - 1) not in visited and x + 1 < maze_size and y - 1 >= 0:
            neighbours.append("right down")
        if (x - 1, y) not in visited and x - 1 >= 0:
            neighbours.append("left")
        if (x - 1, y + 1) not in visited and x - 1 >= 0 and y + 1 < maze_size:
            neighbours.append("left up")
        if (x - 1, y - 1) not in visited and x - 1 >= 0 and y - 1 >= 0:
            neighbours.append("left down")
        if (x, y + 1) not in visited and y + 1 < maze_size:
            neighbours.append("up")
        if (x, y - 1) not in visited and y - 1 >= 0:
            neighbours.append("down")

        # Wylosowanie sąsiada i przejsćie do niego (z otworzeniem ściany)
        if len(neighbours) > 0:
            neighbour_choosen = (random.choice(neighbours))
            if neighbour_choosen == "right":
                my_cells[x, y].open_wall(7)
                my_cells[x + 1, y].open_wall(3)
                x = x + 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "right up":
                my_cells[x, y].open_wall(0)
                my_cells[x + 1, y + 1].open_wall(4)
                x = x + 1
                y = y + 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "right down":
                my_cells[x, y].open_wall(6)
                my_cells[x + 1, y - 1].open_wall(2)
                x = x + 1
                y = y - 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "left":
                my_cells[x, y].open_wall(3)
                my_cells[x - 1, y].open_wall(7)
                x = x - 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "left up":
                my_cells[x, y].open_wall(2)
                my_cells[x - 1, y + 1].open_wall(6)
                x = x - 1
                y = y + 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "left down":
                my_cells[x, y].open_wall(4)
                my_cells[x - 1, y - 1].open_wall(0)
                x = x - 1
                y = y - 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "up":
                my_cells[x, y].open_wall(1)
                my_cells[x, y + 1].open_wall(5)
                y = y + 1
                visited.append((x, y))
                stack.append((x, y))
            elif neighbour_choosen == "down":
                my_cells[x, y].open_wall(5)
                my_cells[x, y - 1].open_wall(1)
                y = y - 1
                visited.append((x, y))
                stack.append((x, y))
        else:
            x, y = stack.pop()  # dgy nie ma nieodwiedzanych sąsaidó odrzucamy komórkę ze stosu

maze_size = 10
cell_array = np.empty((maze_size, maze_size), dtype=object)

for i in range(maze_size):
    for j in range(maze_size):
         cell_array[i][j] = Cell(i, j)

# Tworzenie labiryntu, wejscia i wyjscia
create_maze(cell_array)
cell_array[0, maze_size-1].open_wall(3)
cell_array[maze_size-1][0].open_wall(7)



# Rysowanie
for j in range(maze_size):
    for k in range(maze_size):
        for i in range(len(cell_array[j][k].points)):
            if cell_array[j][k].points[i][0] >= 0 and cell_array[j][k].points[i][1] >= 0:
               plt.plot((cell_array[j][k].points[i][0], abs(cell_array[j][k].points[(i+1)%8][0])),
                        (cell_array[j][k].points[i][1], abs(cell_array[j][k].points[(i+1)%8][1])), 'b')


plt.show()


