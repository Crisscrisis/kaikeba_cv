import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import math

flag = 1
img_ori = cv2.imread('./lenna.jpg',flag)
img_ori = cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)
print(img_ori.shape)
print("img_ori:\n",img_ori)

def image_show(img):
    plt.figure(figsize=(3,3))
    plt.imshow(img)
    plt.show()

def image_crop(img,x1,x2,y1,y2):
    newImg = img[x1:x2,y1:y2]
    return newImg

def color_shift(img,b_shift,g_shift,r_shift):
    b_tmp, g_tmp, r_tmp = cv2.split(img)
    for row in range(b_tmp.shape[0]):
        for col in range(b_tmp.shape[1]):
            if ((b_tmp[row][col] + b_shift) <= 255) and ((b_tmp[row][col] + b_shift) >= 0):
                b_tmp[row][col] = b_tmp[row][col] + b_shift
            elif ((b_tmp[row][col] + b_shift) >= 255):
                b_tmp[row][col] = 255
            else:
                b_tmp[row][col] = 0

    for row in range(g_tmp.shape[0]):
        for col in range(g_tmp.shape[1]):
            if ((g_tmp[row][col] + g_shift) <= 255) and ((g_tmp[row][col] + g_shift) >= 0):
                g_tmp[row][col] = g_tmp[row][col] + g_shift
            elif ((g_tmp[row][col] + g_shift) >= 255):
                g_tmp[row][col] = 255
            else:
                g_tmp[row][col] = 0

    for row in range(r_tmp.shape[0]):
        for col in range(r_tmp.shape[1]):
            if ((r_tmp[row][col] + r_shift) <= 255) and ((r_tmp[row][col] + r_shift) >= 0):
                r_tmp[row][col] = r_tmp[row][col] + r_shift
            elif ((r_tmp[row][col] + r_shift) >= 255):
                r_tmp[row][col] = 255
            else:
                r_tmp[row][col] = 0
    return cv2.merge((b_tmp,g_tmp,r_tmp))

def rotation():
    pass

def perspectice_transform():
    pass


img_shift = color_shift(img_ori,500,-1000,500)
print("img_shift:\n",img_shift)
image_show(img_ori)
image_show(img_shift)

