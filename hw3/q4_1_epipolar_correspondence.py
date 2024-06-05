import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()
    plt.subplots_adjust(right=0.98,left=0.02, bottom = 0, top = 1, wspace=0.01) 

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use gaussian weighting to weight the pixel simlairty

"""


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    window_size = 10
    search_range = 30
    patch1 = im1[y1-window_size:y1+window_size+1,
                 x1-window_size:x1+window_size+1]
    coeff = F@np.array([x1, y1, 1]).T
    coeff = coeff/np.linalg.norm(coeff)
    # print(coeff)
    h, w, _ = im2.shape

    y2_values = np.arange(y1-search_range,y1+search_range)
    x2_values = np.rint(-(coeff[1]*y2_values + coeff[2])/coeff[0])

    mask = (x2_values >= 0) & (x2_values < w)
    y2_values = y2_values[mask].astype(int)
    x2_values = x2_values[mask].astype(int)

    # Gaussian mask, ref:
    # https://www.geeksforgeeks.org/how-to-generate-2-d-gaussian-array-using-numpy/
    sigma, muu = 3, 0
    xgrid, ygrid = np.meshgrid(np.arange(-window_size, window_size+1),
                      np.arange(-window_size, window_size+1))
    dst = np.sqrt(xgrid**2+ygrid**2)
    normal = np.sqrt(2.0 * np.pi*sigma**2)
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))/normal
    gauss = np.sum(gauss)

    x2_best, y2_best = 0, 0
    err_best = float('inf')
    for i in range(y2_values.shape[0]):
        rect = [y2_values[i]-window_size, y2_values[i]+window_size+1, 
                x2_values[i]-window_size, x2_values[i]+window_size+1]

        if rect[0] < 0 or rect[1] >= h or rect[2] < 0 or rect[3] >= w:
            continue
        
        patch2 = im2[rect[0]:rect[1],
                     rect[2]:rect[3]] 
        err = np.linalg.norm((patch1 - patch2) * gauss)
        if err < err_best:
            err_best = err
            x2_best = x2_values[i]
            y2_best = y2_values[i]
    return x2_best, y2_best


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez("q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    print(x2, y2)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
