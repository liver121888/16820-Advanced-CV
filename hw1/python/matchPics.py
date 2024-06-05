import numpy as np
from skimage.color import rgb2gray
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        
        # TODO: Convert Images to GrayScale
        I1_g = rgb2gray(I1)
        I2_g = rgb2gray(I2)
        
        # TODO: Detect Features in Both Images
        locs1 = corner_detection(I1_g, sigma)
        locs2 = corner_detection(I2_g, sigma)

        # TODO: Obtain descriptors for the computed feature locations
        desc1, desc1_locs = computeBrief(I1_g, locs1)
        desc2, desc2_locs = computeBrief(I2_g, locs2)

        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)

        return matches, desc1_locs, desc2_locs
