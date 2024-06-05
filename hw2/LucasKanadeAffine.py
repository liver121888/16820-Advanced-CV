import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ################### TODO Implement Lucas Kanade Affine ###################
    h, w = It.shape[0], It.shape[1]
    xarr = np.linspace(0, w, w, endpoint=False)
    yarr = np.linspace(0, h, h, endpoint=False)
    z = RectBivariateSpline(yarr, xarr, It)
    z1 = RectBivariateSpline(yarr, xarr, It1)
    ygrid, xgrid = np.mgrid[0:h+1:h*1j, 0:w+1:w*1j]
    
    # initial guess
    p = p0
    for _ in range(num_iters):

        # warpAffine not working well
        # warped_ygrid = cv2.warpAffine(ygrid, M, (w, h))
        # warped_xgrid = cv2.warpAffine(xgrid, M, (w, h))
        warped_ygrid = M[1,0]*xgrid + M[1,1]*ygrid + M[1,2]
        warped_xgrid = M[0,0]*xgrid + M[0,1]*ygrid + M[0,2]
        # pdb.set_trace()
        # check common region by warping grid
        # referecce: Sihan Liu
        condition_y = np.logical_and(warped_ygrid > 0, warped_ygrid < h)
        condition_x = np.logical_and(warped_xgrid > 0, warped_xgrid < w)
        common_area = np.logical_and.reduce((condition_y, condition_x))
        
        # pdb.set_trace()
        warped_ygrid = warped_ygrid[common_area]
        warped_xgrid = warped_xgrid[common_area]
        common_ygrid = ygrid[common_area]
        common_xgrid = xgrid[common_area]

        warped_It1 = z1.ev(warped_ygrid, warped_xgrid)
        warped_x_grad = z1.ev(warped_ygrid, warped_xgrid, dy=1)
        warped_y_grad = z1.ev(warped_ygrid, warped_xgrid, dx=1)
        template = z.ev(common_ygrid, common_xgrid)

        length = template.shape[0]

        # compute error image
        b = template - warped_It1

        # cal Jacobian, A should be warped gradient * Jacobian
        A = np.zeros((length, 6))
        A[:, 0] = warped_x_grad * common_xgrid
        A[:, 1] = warped_x_grad * common_ygrid
        A[:, 2] = warped_x_grad 
        A[:, 3] = warped_y_grad * common_xgrid
        A[:, 4] = warped_y_grad * common_ygrid
        A[:, 5] = warped_y_grad

        # pdb.set_trace()
        delta_p = np.dot(np.linalg.inv(np.dot(A.T, A)),np.dot(A.T,b))

        # print("delta_p" + str(delta_p.flatten()))
        if np.linalg.norm(delta_p)**2 < threshold:
            # don't have to move at all, we found the p!
            # print("delta_p < threshold")
            break
        p += delta_p.flatten()
        M[0,0] = p[0] + 1
        M[0,1] = p[1] + 0
        M[0,2] = p[2] + 0
        M[1,0] = p[3] + 0
        M[1,1] = p[4] + 1
        M[1,2] = p[5] + 0
    return M