#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import cv2 as cv

if __name__ == '__main__':

    dst = cv.imread(r'..\pic\left\left.jpg')
    dst2 = cv.imread(r'..\pic\right\right.jpg')
    #转换为灰度图后计算视差图
    imgL = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(dst2, cv.COLOR_BGR2GRAY)

    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv.imshow('disparity', (disp - min_disp) / num_disp)
    cv.waitKey()
    cv.destroyAllWindows()