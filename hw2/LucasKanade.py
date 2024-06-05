import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    xarr = np.linspace(0, It.shape[1], It.shape[1], endpoint=False)
    yarr = np.linspace(0, It.shape[0], It.shape[0], endpoint=False)
    z = RectBivariateSpline(yarr, xarr, It)
    z1 = RectBivariateSpline(yarr, xarr, It1)
    x1, y1, x2, y2 = rect

    ygrid, xgrid = np.mgrid[y1:y2+1:(y2-y1)*1j, x1:x2+1:(x2-x1)*1j]
    template = z.ev(ygrid, xgrid)

    # initial guess
    p = p0
    # It (row, col), rect (x1, y1, x2, y2)
    for _ in range(num_iters):
        new_patch = z1.ev(ygrid+p[1], xgrid+p[0])
        x_grad = z1.ev(ygrid+p[1], xgrid+p[0], dy=1)
        y_grad = z1.ev(ygrid+p[1], xgrid+p[0], dx=1)
        A = np.array([x_grad.flatten(), y_grad.flatten()]).T
        b = np.expand_dims((template - new_patch).flatten(),axis=1)
        delta_p = np.dot(np.linalg.inv(np.dot(A.T, A)),np.dot(A.T,b))
        # p suppose can minimize the error between two patch
        if np.linalg.norm(delta_p)**2 < threshold:
            # don't have to move at all, we found the p!
            # print("delta_p < threshold")
            break
        p += delta_p.flatten()
    return p
