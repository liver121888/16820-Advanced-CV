import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.interpolate import RectBivariateSpline

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    
    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################
    h, w = image1.shape[0], image1.shape[1]
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    # print(M)
    # we calculate LKA in common area, we have to exclude not common area in here
    xarr = np.linspace(0, w, w, endpoint=False)
    yarr = np.linspace(0, h, h, endpoint=False)
    z = RectBivariateSpline(yarr, xarr, image1)
    z1 = RectBivariateSpline(yarr, xarr, image2)
    ygrid, xgrid = np.mgrid[0:h+1:h*1j, 0:w+1:w*1j]
    
    # check common region by warping grid
    warped_xgrid = M[0,0]*xgrid + M[0,1]*ygrid + M[0,2]
    warped_ygrid = M[1,0]*xgrid + M[1,1]*ygrid + M[1,2]

    # pdb.set_trace()
    # check common region by warping grid
    # filter out not common region
    # referecce: Sihan Liu
    condition_y = np.logical_and(warped_ygrid < 0, warped_ygrid >= h)
    condition_x = np.logical_and(warped_xgrid < 0, warped_xgrid >= w)
    not_common = np.logical_and.reduce((condition_y, condition_x))
   
    warped_It1 = z1.ev(warped_ygrid, warped_xgrid)
    It = z.ev(ygrid, xgrid)
    warped_It1[not_common] = 0
    It[not_common] = 0
    # make sure it's common by adding warped_It1 != 0
    pixels = (warped_It1 != 0) & (abs(warped_It1 - It) > tolerance)
    mask[pixels] = 1
    mask = binary_dilation(mask)
    mask = binary_erosion(mask)
    return mask.astype(bool)
