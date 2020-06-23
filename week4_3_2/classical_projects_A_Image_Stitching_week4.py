# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:23:38 2020

@author: 孟京浩
"""
import numpy as np
import cv2


class Stitcher:
    # 拼接函数
    def stitch(self, images, ratio=0.5, thresh=3):
        # 获取输入图片
        (imageB, imageA) = images
        # 检测A、 B 图片的SIFT 关键特征点，并计算特征描述子
        (kpsA, descriptorsA) = self.extract_sift_Features(imageA)
        (kpsB, descriptorsB) = self.extract_sift_Features(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, descriptorsA, descriptorsB, ratio, thresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则， 提取匹配结果
        # H 是3*3 视角变换矩阵
        (matches, H, status) = M
        # 将图片A进行视角变换，result是变换后的图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # 将图片B传入图片最左端
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 去除黑色部分
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        result = result[min_row:max_row, min_col:max_col, :]



        imageC = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        # 返回匹配结果和拼接图
        return (result, imageC)

    # SIFT算法找出图片的关键点集合
    def extract_sift_Features(self, image):
        # 将彩色图片转换成灰度图
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        sift_initialize = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, descriptors) = sift_initialize.detectAndCompute(image_gray, None)

        # 将结果转换成Numpy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，以及对应的描述特征
        return kps, descriptors

    # 匹配两张图片的特征点集合
    def matchKeypoints(self, kpsA, kpsB, descriptorsA, descriptorsB, ratio, thresh):
        # 建立暴力匹配器
        bruteForce = cv2.BFMatcher_create()
        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = bruteForce.knnMatch(descriptorsA, descriptorsB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配度大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, thresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    # 把对应点连线
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        imageC = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        imageC[0:hA, 0:wA] = imageA
        imageC[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点匹配成功时，画到拼接图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(imageC, ptA, ptB, (0, 255, 0), 1)

        # 返回结果
        return imageC

# 读取拼接图片
imageA = cv2.imread("kit1_1.jpg")
imageB = cv2.imread("kit2_1.jpg")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB])

cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
