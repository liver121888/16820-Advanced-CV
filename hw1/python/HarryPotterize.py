import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, computeHCV2, compositeH

# Import necessary functions

# Q2.2.4

def warpImage():

    # step 1
    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')
    image3 = cv2.imread('../data/hp_cover.jpg')
    # fix not filling up problem
    image3 = cv2.resize(image3, (image1.shape[1], image1.shape[0]))

    # step 2
    matches, desc1_locs, desc2_locs = matchPics(image1, image2, opts)
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

    bestH2to1, inliers = computeH_ransac(m1, m2, opts)
    # H, inliers = computeHCV2(m1, m2, cv2.RANSAC)
    print(bestH2to1)
    # print(H)
    composite_img = compositeH(bestH2to1, image3, image2)
    return composite_img

if __name__ == "__main__":
    composite_img = warpImage()
    cv2.imshow('composite_img',composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

