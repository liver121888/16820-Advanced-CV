import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    image = cv2.imread('../data/cv_cover.jpg')
    histogram = []
    num_rot = 36
    loop = [x for x in range(num_rot)]
    for i in loop:
        print(i)

        # TODO: Rotate Image
        rotated = rotate(image, i * 10)

        # TODO: Compute features, descriptors and Match features
        matches, desc1_locs, desc2_locs = matchPics(image, rotated, opts)

        # TODO: Match features using the descriptors
        if i == 0 or i == 1 or i == 15:
            plotMatches(image, rotated, matches, desc1_locs, desc2_locs)
    
        # TODO: Update histogram
        histogram.append(len(matches))

    print(histogram)
    # TODO: Display histogram
    plt.bar(loop,histogram)
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
