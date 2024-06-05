import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors

# Insert your package here
def triangulate2(C1, pts1, C2, pts2):
    err = 0
    x1 = pts1[0]
    y1 = pts1[1]
    x2 = pts2[0]
    y2 = pts2[1]
    A1 = y1*C1[2,:] - C1[1,:]
    A2 = C1[0,:] - x1*C1[2,:]
    A3 = y2*C2[2,:] - C2[1,:]
    A4 = C2[0,:] - x2*C2[2,:]
    A = np.vstack((A1, A2, A3, A4))
    U, S, Vh = np.linalg.svd(A)
    w = Vh[-1]
    w = w/w[-1]
    return w

def triangulate3(C1, pts1, C2, pts2, C3, pts3):
    P = np.zeros((1,3))
    err = 0
    x1 = pts1[0]
    y1 = pts1[1]
    x2 = pts2[0]
    y2 = pts2[1]
    x3 = pts3[0]
    y3 = pts3[1]
    A1 = y1*C1[2,:] - C1[1,:]
    A2 = C1[0,:] - x1*C1[2,:]
    A3 = y2*C2[2,:] - C2[1,:]
    A4 = C2[0,:] - x2*C2[2,:]
    A5 = y3*C3[2,:] - C3[1,:]
    A6 = C3[0,:] - x3*C3[2,:]
    A = np.vstack((A1, A2, A3, A4, A5, A6))
    U, S, Vh = np.linalg.svd(A)
    w = Vh[-1]
    w = w/w[-1]
    return w




"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""

# colaborator: Parth
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    n = pts1.shape[0]
    P = np.zeros((n,4))
    for i in range(pts1.shape[0]):
        p1, p2, p3 = pts1[i], pts2[i], pts3[i]
        print(p1[2], p2[2], p3[2])
        if p1[2] < Thres or p2[2] < Thres or p3[2] < Thres:
            continue
        if p1[2] > Thres and p2[2] > Thres and p3[2] > Thres:
            p = triangulate3(C1, p1, C2, p2, C3, p3)
        elif p1[2] > Thres and p2[2] > Thres:
            p = triangulate2(C1, p1, C2, p2)
        elif p1[2] > Thres and p3[2] > Thres:
            p = triangulate2(C1, p1, C3, p3)
        elif p2[2] > Thres and p3[2] > Thres:
            p = triangulate2(C2, p2, C3, p3)
        P[i] = p

    proj1 = (C1@P.T).T
    proj1 = proj1/np.expand_dims(proj1[:, -1],axis=1)
    proj2 = (C2@P.T).T
    proj2 = proj2/np.expand_dims(proj2[:, -1],axis=1)
    proj3 = (C3@P.T).T
    proj3 = proj3/np.expand_dims(proj3[:, -1],axis=1)
    print(pts1.shape)
    print(proj1.shape)
    err = np.linalg.norm(pts1[:,:2] - proj1[:,:2])**2
    + np.linalg.norm(pts2[:,:2] - proj2[:,:2])**2
    + np.linalg.norm(pts3[:,:2] - proj3[:,:2])**2
    return P, err


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    connections_3d = [[0, 1],[1, 3],[2, 3],[2, 0],
        [4, 5],[6, 7],[8, 9],[9, 11],
        [10, 11],[10, 8],[0, 4],[4, 8],
        [1, 5],[5, 9],[2, 6],[6, 10],[3, 7],[7, 11],]
    colors = ["blue","blue","blue","blue",
        "red","magenta","green","green","green","green","red","red",
        "red","red","magenta","magenta","magenta","magenta",]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(pts_3d_video)):
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d_video[i][index0, 0], pts_3d_video[i][index1, 0]]
            yline = [pts_3d_video[i][index0, 1], pts_3d_video[i][index1, 1]]
            zline = [pts_3d_video[i][index0, 2], pts_3d_video[i][index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])
    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()
    return


# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]
        print(pts1.shape, pts2.shape, pts3.shape)

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further

        # img1 = visualize_keypoints(im1, pts1)
        # img2 = visualize_keypoints(im2, pts2)
        # img3 = visualize_keypoints(im3, pts3)

        C1 = K1@M1
        C2 = K2@M2
        C3 = K3@M3

        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, 100)
        plot_3d_keypoint(P)
        print(err)
        pts_3d_video.append(P)
    np.savez("q6_1.npz", pts_3d_video)
    plot_3d_keypoint_video(pts_3d_video)