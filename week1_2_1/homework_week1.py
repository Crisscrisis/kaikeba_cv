'''
author      :   Lucas Liu
date        :   2020/1/18
description :   week 1 homework, low level image process method, including crop, color shift,
                rotation, perspective transform, and related testcase
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

'''
description:
    show the picture of the image
input:
    img             image in cv2.imread format(BGR)
output :
    None  
'''
def image_show(img, name):
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(name)
    plt.show()


'''
description:
    crop the image into the given shape
input:
    img             image in cv2.imread format(BGR)
    x1              start of the cropped width
    x2              end of the cropped width
    y1              start of the cropped height
    y2              end of the cropped height
output :
    img_cropped     cropped image
'''
def image_crop(img, x1, x2, y1, y2):
    h_max, w_max = img.shape[:2]
    if x1 >= x2 or y1 >= y2:
        print('x1 should be less than x2, y1 should be less than y2! return origin image')
        return img
    if x1 >= 0 and x2 <= w_max and y1 >= 0 and y2 <= h_max:
        img_cropped = img[x1:x2, y1:y2]
        return img_cropped
    elif x1 < 0 or x2 > w_max:
        print('wrong input in width! return origin image')
        return img
    else:
        print('wrong input in height! return origin image')
        return img


'''
description:
    color shift of the given image in 3 channels
input:
    img             image in cv2.imread format(BGR)
    b_shift         shifted value in blue channel(negative value is accepted)
    g_shift         shifted value in green channel(negative value is accepted)
    r_shift         shifted value in red channel(negative value is accepted)
output :
    img_shifted     color shifted image
'''
def color_shift(img, b_shift, g_shift, r_shift):
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
    img_shifted = cv2.cv2.merge((b_tmp, g_tmp, r_tmp))
    return img_shifted


'''
description:
    rotate the given image in center of the central point
input:
    img             image in cv2.imread format(BGR)
    angle           Counterclockwise rotate angle 
    scale           zoom scale
output :
    img_rotated rotated image
'''
def rotation(img, angle, scale):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img_rotated = cv2.warpAffine(img, M, (w, h))
    return img_rotated


'''
description:
    rotate the given image in center of the central point
input:
    img             image in cv2.imread format(BGR)
    ori_pix         ori_pix in x1, y1, x2, y2, x3, y3, x4, y4
    dst_pix         dst_pix in dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4
output :
    M_warp          warp matrix
    img_wrap        wrapped image
'''
def perspectice_transform(img, ori_pix, dst_pix):
    height, width = img.shape[:2]
    x1, y1, x2, y2, x3, y3, x4, y4 = ori_pix
    dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = dst_pix
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp


# initialize
flag = 1    # gray:0
img_ori = cv2.imread('./lenna.jpg', flag)
img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
image_show(img_ori, 'img_ori')

# crop test
'''
img_cropped = image_crop(img_ori, 100, 300, 100, 200)   # normal case
img_cropped = image_crop(img_ori, 200, 100, 300, 200)   # exception: x1 > x2
img_cropped = image_crop(img_ori, 100, 100, 200, 200)   # exception: x1 = x2
img_cropped = image_crop(img_ori, -10, 100, 800, 200)   # exception: exceed range
image_show(img_cropped, 'img_cropped')
'''


# color shift test
'''
img_shift = color_shift(img_ori,80,90,70)   # normal case
img_shift = color_shift(img_ori,500,-10,700)    # exception: exceed range [0, 255]
print("img_shift:\n",img_shift)
image_show(img_shift, 'img_shift')
'''


# rotation test
'''
img_rotated = rotation(img_ori, 45, 2)  # noamal case: counter clockwise
img_rotated = rotation(img_ori, -20, 2)  # noamal case: clockwise
image_show(img_rotated, 'img_rotated')
'''


# perspective_transform test
'''
height, width, channels = img_ori.shape
random_margin = 600
x1 = random.randint(-random_margin, random_margin)
y1 = random.randint(-random_margin, random_margin)
x2 = random.randint(width - random_margin - 1, width - 1)
y2 = random.randint(-random_margin, random_margin)
x3 = random.randint(width - random_margin - 1, width - 1)
y3 = random.randint(height - random_margin - 1, height - 1)
x4 = random.randint(-random_margin, random_margin)
y4 = random.randint(height - random_margin - 1, height - 1)

dx1 = random.randint(-random_margin, random_margin)
dy1 = random.randint(-random_margin, random_margin)
dx2 = random.randint(width - random_margin - 1, width - 1)
dy2 = random.randint(-random_margin, random_margin)
dx3 = random.randint(width - random_margin - 1, width - 1)
dy3 = random.randint(height - random_margin - 1, height - 1)
dx4 = random.randint(-random_margin, random_margin)
dy4 = random.randint(height - random_margin - 1, height - 1)
ori_pix = (x1, y1, x2, y2, x3, y3, x4, y4)
dst_pix = (dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)
M_warp, img_persp = perspectice_transform(img_ori, ori_pix, dst_pix)
image_show(img_persp, 'img_persp')
'''
