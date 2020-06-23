'''
author      :   Lucas Liu
date        :   2020/2/24
description :   week 4 project 1 --- B: object detection
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

MIN_MATCH_COUNT = 20 # at least 20 matches are to be there to find the object

img1 = cv2.imread('card.png', 0)          # queryImage
img2 = cv2.imread('card_in_scene.png', 0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # use algorithm: SIFT
search_params = dict(checks = 100) # the trees in the index should be recursively traversed 100 times

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2) # extract locations of mathced keypoints in queryImage
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2) # extract locations of mathced keypoints in trainImage

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # find homography between 2 image
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M) # find perpective transformation

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

plt.title('object detection: queryImage')
img1rgb = cv2.imread('card.png', 1)
img1rgb = cv2.cvtColor(img1rgb, cv2.COLOR_BGR2RGB)
plt.imshow(img1rgb), plt.show()

plt.title('object detection: trainImage')
img2rgb = cv2.imread('card_in_scene.png', 1)
img2rgb = cv2.cvtColor(img2rgb, cv2.COLOR_BGR2RGB)
plt.imshow(img2rgb), plt.show()

plt.title('object detection: find object(queryImage) in trainImage')
plt.imshow(img3, 'gray'), plt.show()
