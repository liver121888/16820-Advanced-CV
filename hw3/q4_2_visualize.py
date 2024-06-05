import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


"""
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2200 on the 3D points. 

    Modified by Vineet Tambe, 2023.
"""


def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):
    # ----- TODO -----
    pts2 = np.zeros(temple_pts1.shape)
    for i in range(temple_pts1.shape[0]):
        pts2[i] = epipolarCorrespondence(im1, im2, F, 
                    temple_pts1[i, 0], temple_pts1[i, 1])
    M2, C2, P = findM2(F, temple_pts1, pts2, intrinsics, "q4_2.npz")
    return P

def plot_3D(P):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    plt.subplots_adjust(right=0.98,left=0.02, bottom = 0, top = 1, wspace=0.01) 
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()

"""
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
"""
if __name__ == "__main__":
    temple_coords = np.load("data/templeCoords.npz")
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # Call compute3D_pts to get the 3D points and visualize using matplotlib scatter
    temple_pts1 = np.hstack([temple_coords["x1"], temple_coords["y1"]])

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    # ax1.imshow(im1)
    # ax1.set_axis_off()
    # ax1.plot(pts1[:,0], pts1[:,1], "*", markersize=3, linewidth=2)
    # ax2.imshow(im2)
    # ax2.plot(pts2[:,0], pts2[:,1], "ro", markersize=3, linewidth=2)
    # plt.show()
    M1 = np.hstack((np.eye(3),np.zeros((3,1))))
    C1 = K1@M1
    np.savez("q4_2.npz", F, C1)

    P = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)

    # print(P)
    # print(P.max())

    # Visualize
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_3D(P)


