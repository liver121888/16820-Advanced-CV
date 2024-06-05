import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
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
    U, S, Vh = np.linalg.svd(A)
    F1 = Vh[-2].reshape((3,3))
    F2 = Vh[-1].reshape((3,3))
    F_mat = [F1, F2]
    # print(F_mat[0].shape)
    D = np.zeros((2,2,2))
    D_tmp = np.zeros((3,3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                D_tmp[:,0] = F_mat[i][:,0]
                D_tmp[:,1] = F_mat[j][:,1]
                D_tmp[:,2] = F_mat[k][:,2]
                D[i,j,k] = np.linalg.det(D_tmp)
    coefficients = []
    coefficients.append(-D[1,0,0]+D[0,1,1]+D[0,0,0]+D[1,1,0]+
                        D[1,0,1]-D[0,1,0]-D[0,0,1]-D[1,1,1])
    coefficients.append(D[0,0,1]-2*D[0,1,1]-2*D[1,0,1]+D[1,0,0]
                        -2*D[1,1,0]+D[0,1,0]+3*D[1,1,1])
    coefficients.append(D[1,1,0]+D[0,1,1]+D[1,0,1]-3*D[1,1,1])
    coefficients.append(D[1,1,1])
    alphas = np.roots(coefficients)
    for alpha in alphas:
        F = alpha * F1 + (1 - alpha) * F2
        F = refineF(F,p1_norm,p2_norm)
        # unscaling
        F = T.T@F@T
        # F = F/F[2,2]
        Farray.append(F)  
    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)
    # print(Farray)
    # F = Farray[1]

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        print('=====iter ' + str(i) + '=====' )
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    F = F/F[2,2]
    print("F: " + np.array2string(F))
    print("Error:", ress[min_idx])

    displayEpipolarF(im1, im2, F)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
    np.savez("q2_2.npz", F, M)