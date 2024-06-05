import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2
from q4_2_visualize import plot_3D

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    plt.subplots_adjust(right=0.98,left=0.02, bottom = 0, top = 1, wspace=0.01) 
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # Replace pass by your implementation
    p1_homo, p2_homo = toHomogenous(pts1), toHomogenous(pts2)
    inliers_best_cnt = 0
    for _ in range(nIters):
    # for _ in range(1):
        idxs = np.random.choice(pts1.shape[0], 8, replace=False)
        p1_pick = pts1[idxs]
        p2_pick = pts2[idxs]
        F = eightpoint(p1_pick, p2_pick, M)        
        err = calc_epi_error(p1_homo, p2_homo, F)
        inliers_mask = err < tol
        masked_indices = np.where(inliers_mask)
        inliers_cnt = (p2_homo[masked_indices]).shape[0]
        if inliers_cnt > inliers_best_cnt:
            best_F = F
            inliers_best_cnt = inliers_cnt
            inliers = inliers_mask
    # should around 106
    print("inliers_best_cnt: ", inliers_best_cnt)

    p1_inliers = pts1[inliers]
    p2_inliers = pts2[inliers]
    best_F = eightpoint(p1_inliers, p2_inliers, M) 

    return best_F, inliers

"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    u = r/theta
    u1, u2, u3 = u
    u_cross = np.array([[0, -u3, u2],
                        [u3, 0, -u1],
                        [-u2, u1, 0]])
    u = np.expand_dims(u,axis=-1)
    R = np.eye(3)*np.cos(theta)+(1-np.cos(theta))*np.dot(u, u.T) + \
        u_cross*np.sin(theta)
    return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    A = (R-R.T)/2
    rho = np.array([A[2,1], A[0,2], A[1,0]])
    s = np.linalg.norm(rho)
    c = (R[0,0] + R[1,1] + R[2,2] - 1)/2
    float_cmp = 1e-3
    if s < float_cmp and abs(c - 1) < float_cmp:
        return np.zeros((1,3))
    elif s < float_cmp and abs(c + 1) < float_cmp:
        RI = R + np.eye(3)
        for i in range(RI.shape[-1]):
            if np.count_nonzero(RI[:,i]) > 0:
                v = RI[:,i]
                break
        u = v/np.linalg.norm(v)
        r = u*np.pi
        r1, r2, r3 = r
        if (np.linalg.norm(r) == np.pi) and \
            ((r1 == r2 == 0 and r3 < 0) or \
             (r1 == 0 and r2 < 0) or (r1 < 0)):
            return -r
        else:
            return r
    else:
        u = rho/s
        theta = np.arctan2(s,c)
        r = u*theta
        return r
    
"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""

def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    P, r2, t2 = np.split(x,[-6,-3])
    P = P.reshape((-1, 3))
    R2 = rodrigues(r2)
    C1 = K1@M1
    C2 = K2@np.hstack((R2.reshape((3,3)),  t2.reshape((3,1))))
    P_homo = np.hstack((P, np.ones((P.shape[0], 1))))
    p1_hat_homo = C1@P_homo.T
    p2_hat_homo = C2@P_homo.T
    p1_hat = (p1_hat_homo/p1_hat_homo[-1,:])[:2,:]
    p2_hat = (p2_hat_homo/p2_hat_homo[-1,:])[:2,:]
    residuals = np.concatenate([(p1-p1_hat.T).reshape(-1), 
                                (p2-p2_hat.T).reshape(-1)])
    return residuals


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # print(M2_init)
    R2, t2 = M2_init[:3, :3], M2_init[:,-1]
    # print(R2,t2)
    r2 = invRodrigues(R2)

    # x, the flattened concatenationg of P, r2, and t2.
    # P_init.shape = 140, 3, r2.shape = (3, ), t2.shape = (3, )
    x = np.concatenate((P_init.flatten(), r2, t2))

    obj_start = rodriguesResidual(K1, M1, p1, K2, p2, x)
    print("obj_start", np.linalg.norm(obj_start)**2)

    def objfun(x_sub): 
        val = np.linalg.norm(rodriguesResidual(K1, M1, p1, K2, p2, x_sub))**2
        return val

    res = scipy.optimize.minimize(objfun, x,method='BFGS', 
                                  options={'maxiterint': 10000, 'disp': True})
    
    P_best, r2_best, t2_best = np.split(res.x,[P_init.flatten().shape[0], 
                    P_init.flatten().shape[0]+r2.shape[0]])

    P_best = P_best.reshape(-1,3)
    x_best = np.concatenate((P_best.flatten(), r2_best, t2_best))

    obj_end = rodriguesResidual(K1, M1, p1, K2, p2, x_best)
    print("obj_end",  np.linalg.norm(obj_end)**2)

    R2_best = rodrigues(r2_best)
    M2 = np.hstack((R2_best, t2_best.reshape((3, 1))))

    return M2, P_best, obj_start, obj_end


if __name__ == "__main__":
    # np.random.seed(1)  # Added for testing, can be commented out

    # some_corresp_noisy = np.load(
    #     "data/some_corresp_noisy.npz"
    # )  # Loading correspondences
    # intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    # K1, K2 = intrinsics["K1"], intrinsics["K2"]
    # noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    # im1 = plt.imread("data/im1.png")
    # im2 = plt.imread("data/im2.png")

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))

    # # displayEpipolarF(im1, im2, F)

    # # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    # assert F.shape == (3, 3)
    # assert F[2, 2] == 1
    # assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    # rotVec = sRot.from_matrix([[0, -1, 0],
    #                             [1, 0, 0],
    #                             [0, 0, 1]])
    # print("rotVec", rotVec.as_rotvec())
    mat = rodrigues(rotVec.as_rotvec())
    # print("mat", mat)
    r = invRodrigues(mat)
    # print("r", r)

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """

    # F = eightpoint(pts1, pts2, M)
    # displayEpipolarF(im1, im2, F)

    F, inliers = ransacF(pts1, pts2, M, nIters=200, tol=3.0)
    # displayEpipolarF(im1, im2, F)

    M2, C2, P = findM2(F, pts1[inliers], pts2[inliers], intrinsics, "")

    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    M2_adj, P_adj, obj_start, obj_end = bundleAdjustment(K1, M1, pts1[inliers], K2, M2, pts2[inliers], P)
    plot_3D_dual(P, P_adj)
