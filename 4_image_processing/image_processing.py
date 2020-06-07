from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

img = Image.open("mountain2.jpg")

img.show()
rgb_hist = img.histogram()


x_list = list(range(0, 256))
print(len(rgb_hist))

r_h = rgb_hist[0:256]
g_h = rgb_hist[256:512]
b_h = rgb_hist[512:256*3]

fig_b = plt.bar(x_list, r_h, color='r')
plt.title("Red")

plt.figure()
plt.bar(x_list, g_h, color='g')
plt.title("Green")

plt.figure()
plt.bar(x_list, b_h, color='b')
plt.title("Blue")

#GRAY SCALE
#------------------------

img = img.convert("L")
img.show()


grayscale_hist = img.histogram()

plt.figure()
plt.bar(x_list, grayscale_hist)
plt.title("Grayscale")
#------------------------




my_mask = np.array([[0, 2, 2],
                    [-2, 0, 2],
                    [-2, -2, 0]])
my_mask = my_mask.flatten()

my_filter = ImageFilter.Kernel((3,3),my_mask, 1, 0)
im_mf = img.filter(filter=my_filter)

im_mf.show()


plt.show()