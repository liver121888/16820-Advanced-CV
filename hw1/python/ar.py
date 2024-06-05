import numpy as np
import cv2
import pdb

# Import necessary functions
from helper import loadVid
from opts import get_opts
from matchPics import matchPics
from planarH import computeHCV2, compositeH, computeH_ransac
from displayMatch import displayMatched

#Write script for Q3.1
if __name__ == "__main__":
    opts = get_opts()
    # TODO: test with 0.05
    opts.sigma = 0.15
    opts.ratio = 0.7
    opts.max_iters = 500
    opts.inlier_tol = 2.0

    src = loadVid('../data/ar_source.mov')
    dist = loadVid('../data/book.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    # height, width
    src_shape = src[0].shape
    crop_shape = (src_shape[0], int(cv_cover.shape[1]*src_shape[0]/cv_cover.shape[0]))
    v_shape = dist[0].shape
    # print(src.shape): (511, 360, 640, 3)
    # print(dist.shape): (641, 480, 640, 3)
    v_len = min(src.shape[0], dist.shape[0])
    v_len_ratio = 1
    fps = 30
    output = []
    import datetime
    current_dateTime = datetime.datetime.now()
    import os
    param = current_dateTime.strftime('%S%M%H%m%d%Y') + 'p' + str(int(opts.sigma*1000)) + 's' + \
        str(int(opts.ratio*100)) + 'r' + str(opts.max_iters) + \
    'max' + str(opts.inlier_tol) + 'tol'
    dirname = "../data/" + param + "/"
    os.mkdir(dirname)

    # 435,437 prone to get problems, specialize parameters for them
    for i in range(0, v_len//v_len_ratio):
        if i%30 == 0:
            print(i)
        elif i == 395:
            opts.sigma = 0.106
            opts.ratio = 0.8
            opts.max_iters = 1000
            opts.inlier_tol = 2.0
        elif i == 401:
            opts.sigma = 0.15
            opts.ratio = 0.7
            opts.max_iters = 500
            opts.inlier_tol = 2.0
        elif i == 435:
            opts.sigma = 0.106
            opts.ratio = 0.8
            opts.max_iters = 2000
            opts.inlier_tol = 2.0
        elif i == 437:
            opts.sigma = 0.105
        elif i == 439:
            opts.sigma = 0.15
            opts.ratio = 0.7
            opts.max_iters = 500
            opts.inlier_tol = 2.0


        # compute H
        dst_frame = dist[i]
        src_frame = src[i]
        matches, desc1_locs, desc2_locs = matchPics(cv_cover, dst_frame, opts)
        m1 = []
        m2 = []
        for match in matches:
            m1.append(desc1_locs[match[0]])
            m2.append(desc2_locs[match[1]])
        m1 = np.array(m1)
        m2 = np.array(m2)
        # swap x, y
        m1[:, [1, 0]] = m1[:, [0, 1]]
        m2[:, [1, 0]] = m2[:, [0, 1]]
        # bestH2to1, inliers = computeHCV2(m1, m2, cv2.RANSAC)
        bestH2to1, inliers = computeH_ransac(m1, m2, opts)

        # crop source video, h,w = 360 * 286
        cropped = src_frame[:,int(src_shape[1]/2 - crop_shape[1]/2):int(src_shape[1]/2 + crop_shape[1]/2)]
        # fix not filling up problem
        cropped = cv2.resize(cropped, (cv_cover.shape[1], cv_cover.shape[0]))
        # cv2.imshow('out', cropped)
        # cv2.imshow('dst', dst_frame)
        # cv2.waitKey(0)
        composite_img = compositeH(bestH2to1, cropped, dst_frame)
        output.append(composite_img)
        cv2.imwrite(os.path.join(dirname, '{}.jpg'.format(i)), composite_img)

    # reference: https://www.geeksforgeeks.org/saving-a-video-using-opencv/
    # width, height
    result = cv2.VideoWriter('../result/ar.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, (v_shape[1], v_shape[0]))
    for frame in output:
        result.write(frame)
    result.release()

