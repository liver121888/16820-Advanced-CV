import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement InverseCompositionAffine ###################
    h, w = It.shape[0], It.shape[1]
    xarr = np.linspace(0, w, w, endpoint=False)
    yarr = np.linspace(0, h, h, endpoint=False)
    z = RectBivariateSpline(yarr, xarr, It)
    z1 = RectBivariateSpline(yarr, xarr, It1)
    ygrid, xgrid = np.mgrid[0:h+1:h*1j, 0:w+1:w*1j]
    template_x_grad = z.ev(ygrid, xgrid, dy=1)
    template_y_grad = z.ev(ygrid, xgrid, dx=1)

    # cal Jacobian, A should be warped gradient * Jacobian
    A = np.zeros((h*w, 6))
    A[:, 0] = template_x_grad.flatten() * xgrid.flatten()
    A[:, 1] = template_x_grad.flatten() * ygrid.flatten()
    A[:, 2] = template_x_grad.flatten() 
    A[:, 3] = template_y_grad.flatten() * xgrid.flatten()
    A[:, 4] = template_y_grad.flatten() * ygrid.flatten()
    A[:, 5] = template_y_grad.flatten()

    Hessian = np.dot(A.T, A)
    
    # initial guess
    M = M0
    for _ in range(num_iters):

        warped_ygrid = M[1,0]*xgrid + M[1,1]*ygrid + M[1,2]
        warped_xgrid = M[0,0]*xgrid + M[0,1]*ygrid + M[0,2]

        warped_It1 = z1.ev(warped_ygrid, warped_xgrid)
        template = z.ev(xgrid, ygrid)

        # compute error image
        b = (warped_It1 - template).flatten()
       
        # pdb.set_trace()
        delta_p = np.dot(np.linalg.inv(Hessian),np.dot(A.T,b))

        # print("delta_p" + str(delta_p.flatten()))
        if np.linalg.norm(delta_p)**2 < threshold:
            # don't have to move at all, we found the p!
            # print("delta_p < threshold")
            break

        delta_p = delta_p.flatten()
        delta_M = np.zeros((3,3))
        delta_M[0,0] = delta_p[0] + 1
        delta_M[0,1] = delta_p[1] + 0
        delta_M[0,2] = delta_p[2] + 0
        delta_M[1,0] = delta_p[3] + 0
        delta_M[1,1] = delta_p[4] + 1
        delta_M[1,2] = delta_p[5] + 0
        delta_M[2,2] = 1
        # pdb.set_trace()
        M = np.dot(M,np.linalg.inv(delta_M))

    return M
