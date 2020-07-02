```python
'''
author      :   Lucas Liu
date        :   2020/1/23
description :   Pseudo code of RANSAC algorithm in CV(find homograph 
				matrix between points set A and B)
'''

'''
description:
    find homograph matrix between points set A and B
input:
    A           list of list(points set A)
    B           list of list(points set B)
output :
    homoMatrix   final homograph matrix between points set A and B
'''

def ransacMatching(A, B):
# 1.choose 4 pair of points randomly in matching points
numPairs = len(A)
inlierIdx = []
for i in range(4):
    inlierIdx.append((int)(random.random() * numPairs))
inlierPairs = []
for i in range(4):
    inlierPairs.append((A[inlierIdx[i], B[inlierIdx[i]])

epsilon = 0.1 # threshold
cnt = 0

# 5. recursion
while ((bias2 - bias1 != 0) or (cnt != k)):

    # 2.get the homography of the inliers
    homoMatrix = calHomo(inlierParis)
    bias1 = applyHomo(homoMatrix, inlierPairs)

    # 4.get all inliers
    for i in range(numParis):
        if (i not in inlierIdx):

            # 3.apply computed homography matrix
            bias2 = applyHomo(homoMatrix, (A[i], B[i]))
            if (bias2 < epsilon):
                inlierIdx.append(i)
                inlierPairs.append(A[i], B[i])

return homoMatrix
```