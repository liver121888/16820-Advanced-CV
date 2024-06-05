import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here


"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # ----- TODO -----
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    # pts1: (x,y)
    p1_norm = pts1/M
    p2_norm = pts2/M
    n = pts1.shape[0]
    x1 = p1_norm[:,0]
    y1 = p1_norm[:,1]
    x2 = p2_norm[:,0]
    y2 = p2_norm[:,1]
    last_column = np.ones((n))
    # x_prime should be x1, x_prime is from the img with point
    A = np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, last_column)).T
    # print(A.shape)
    U, S, Vh = np.linalg.svd(A)
    # we want V's last column, equivalent to Vh's last row
    F = refineF(Vh[-1],p1_norm,p2_norm)
    # unscaling
    F = T.T@F@T
    F = F/F[2,2]
    return F


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    # print(pts1.shape)
    # (110, 2)
    # print(im1.shape)
    # (480 640 3)
    # print(im2.shape)
    # (480 640 3)
  
    # im1: the img with point, im2: the img with line
    M = np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M)

    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
    print(F, M)

    np.savez("q2_1.npz", F, M)