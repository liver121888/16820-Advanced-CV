import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix


# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    n = pts1.shape[0]
    P = np.zeros((n,3))
    err = 0
    for i in range(n):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
        A1 = y1*C1[2,:] - C1[1,:]
        A2 = C1[0,:] - x1*C1[2,:]
        A3 = y2*C2[2,:] - C2[1,:]
        A4 = C2[0,:] - x2*C2[2,:]
        A = np.vstack((A1, A2, A3, A4))
        U, S, Vh = np.linalg.svd(A)
        w = Vh[-1]
        w = w/w[-1]
        P[i] = w[:3]
        proj1 = C1@w
        proj1 = proj1/proj1[-1]
        proj2 = C2@w
        proj2 = proj2/proj2[-1]
        err += np.linalg.norm(pts1[i] - proj1[:2])**2 + \
            np.linalg.norm(pts2[i] - proj2[:2])**2
    return P, err


"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    # K1: 3*3, M1: 3*4
    K1 = K1/K1[2,2]
    K2 = K2/K2[2,2]
    C1 = K1@np.hstack((np.eye(3),np.zeros((3,1))))
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    best_error = float('inf')
    for i in range(M2s.shape[-1]):
        M2 = M2s[:,:,i]
        C2 = K2@M2
        P, err = triangulate(C1, pts1, C2, pts2)
        if err < best_error and np.min(P[:, 2])>=0:
            best_error = err
            best_M2 = M2
            best_P = P
            best_C2 = C2
    if filename != "":
        np.savez(filename, M2, C2, P)
    return best_M2, best_C2, best_P


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)
    print(M2)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))

    K1 = K1/K1[2,2]
    K2 = K2/K2[2,2]

    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert err < 500
