import numpy as np
import cv2
from helper import computeBrief
from helper import briefMatch
from opts import get_opts
from displayMatch import plotMatches
from matchPics import matchPics
from planarH import computeHCV2, compositeH, computeH_ransac
from skimage.feature import match_descriptors, match_descriptors, SIFT 
from skimage.transform import warp
import pdb

opts = get_opts()
ratio = opts.ratio
left = cv2.imread('../data/pano_left.jpg')
right = cv2.imread('../data/pano_right.jpg')
# left = cv2.imread('../data/pano_left_scaled.jpg')
# right = cv2.imread('../data/pano_right_scaled.jpg')

# reference: 
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html
# https://scikit-image.org/docs/stable/auto_examples/registration/plot_stitching.html

lg = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
rg = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)

descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(lg)
locsl = descriptor_extractor.keypoints
descl = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(rg)
locsr = descriptor_extractor.keypoints
descr = descriptor_extractor.descriptors

# TODO: Match features using the descriptors
matches = match_descriptors(descl,descr,cross_check=True,max_ratio=0.5)
plotMatches(left, right, matches, locsl, locsr)

m1 = []
m2 = []
for match in matches:
    m1.append(locsl[match[0]])
    m2.append(locsr[match[1]])
m1 = np.array(m1)
m2 = np.array(m2)
# swap x, y
m1[:, [1, 0]] = m1[:, [0, 1]]
m2[:, [1, 0]] = m2[:, [0, 1]]
# we want stitch r to l
# H, _ = computeHCV2(m2, m1, cv2.RANSAC)
H, inliers = computeH_ransac(m2, m1, opts)
# create padding
width, height, _ = left.shape
marginh = height
marginw = width
out_shape = height +  marginh, width + marginw
glob_trfm = np.eye(3)
glob_trfm[:2, 2] = marginw/6, marginh/6

left_warp = cv2.warpPerspective(left, glob_trfm, out_shape, flags=cv2.INTER_LINEAR)
# cv2.imshow('left_warp', left_warp)
right_warp = cv2.warpPerspective(right, H, out_shape, flags=cv2.INTER_LINEAR)
# cv2.imshow('right', right_warp)
# cv2.waitKey(0)
right_H = cv2.warpPerspective(right_warp, glob_trfm, out_shape, flags=cv2.INTER_LINEAR)
# cv2.imshow('right', right_H)
# cv2.waitKey(0)
# self-process image addition to solve intensity difference
out = np.zeros(shape=(out_shape[1], out_shape[0], 3), dtype=np.uint8)
for r in range(left_warp.shape[0]):
    for c in range(left_warp.shape[1]):
        if right_H[r, c].any() != 0:
            out[r, c] = right_H[r, c]
        else:
            out[r, c] = left_warp[r, c]
cv2.imshow('panorama', out)
cv2.waitKey(0)
cv2.imwrite('../data/panorama.jpg', out)
cv2.destroyAllWindows()