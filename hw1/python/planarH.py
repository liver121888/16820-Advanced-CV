import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points
    # print(x1.shape[0])

    A_s = [0] * x1.shape[0]
    for i in range(x1.shape[0]):
        A_s[i] = np.array([[x2[i, 0], x2[i, 1], 1, 0, 0, 0, 
                            -x1[i, 0]*x2[i, 0], -x1[i, 0]*x2[i, 1], -x1[i, 0]],
                        [0, 0, 0, x2[i, 0], x2[i, 1], 1,  
                         -x1[i, 1]*x2[i, 0], -x1[i, 1]*x2[i, 1], -x1[i, 1]]])
    A = np.array(A_s)
    A = np.concatenate(A, axis=0)
    # print(A.shape)
    # print(A)
    # print(A.shape)

    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    # print(np.shape(Vh))
    # print(Vh.T)
    # print(Vh.T)
    H = Vh[-1].reshape(3,3)
    H = H/H[2, 2]
    return H


def computeH_norm(x1, x2):
    # suppose x1, x2 are of shape(4, 2)
    # Q2.2.2
    # TODO: Compute the centroid of the points
    c1 = np.sum(x1, axis=0)/x1.shape[0]
    c2 = np.sum(x2, axis=0)/x2.shape[0]

    # TODO: Shift the origin of the points to the centroid
    x1_ori = x1 - c1
    x2_ori = x2 - c2

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x_square_sum = 0
    y_square_sum = 0
    for x in x1_ori:
        x_square_sum += x[0]**2
        y_square_sum += x[1]**2
    s_x_x1 = np.sqrt(2)/np.sqrt(x_square_sum)
    s_y_x1 = np.sqrt(2)/np.sqrt(y_square_sum)

    x_square_sum = 0
    y_square_sum = 0
    for x in x2_ori:
        x_square_sum += x[0]**2
        y_square_sum += x[1]**2
    s_x_x2 = np.sqrt(2)/np.sqrt(x_square_sum)
    s_y_x2 = np.sqrt(2)/np.sqrt(y_square_sum)

    # TODO: Similarity transform
    t1 = np.array([[s_x_x1, 0, -s_x_x1*c1[0]], [0, s_y_x1, -s_y_x1*c1[1]], [0, 0, 1]])
    t2 = np.array([[s_x_x2, 0, -s_x_x2*c2[0]], [0, s_y_x2, -s_y_x2*c2[1]], [0, 0, 1]])

    x1_ori[:, 0] = x1_ori[:, 0]*s_x_x1
    x1_ori[:, 1] = x1_ori[:, 1]*s_y_x1

    x2_ori[:, 0] = x2_ori[:, 0]*s_x_x2
    x2_ori[:, 1] = x2_ori[:, 1]*s_y_x2

    H_norm = computeH(x2_ori, x1_ori)
    # H, inliers = computeHCV2(x1_ori, x2_ori, 0)

    # TODO: Denormalization
    H2to1 = np.dot(np.linalg.inv(t2), np.dot(H_norm, t1))
    H2to1 = H2to1/H2to1[2,2]
    return H2to1

def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
            
    locs1 = np.append(locs1, np.array([[1] for _ in range(locs1.shape[0])]), axis=1)
    locs2 = np.append(locs2, np.array([[1] for _ in range(locs1.shape[0])]), axis=1)

    max_iters = opts.max_iters
    # max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    # dummy values
    H_best = 0
    inliers_best = 0
    inliers_best_cnt = 0
    pick = np.floor(locs1.shape[0] * np.random.rand(max_iters,4))
    same_cnt = 0
    for i in range(max_iters):
        o = int(pick[i, 0])
        p = int(pick[i, 1])
        q = int(pick[i, 2])
        r = int(pick[i, 3])
        if o == p or o == q or o == r or p == q or p == r or q == r:
            same_cnt += 1
            continue

        l1 = np.array([locs1[o,:2], locs1[p,:2],  locs1[q,:2], locs1[r,:2]])
        l2 = np.array([locs2[o,:2], locs2[p,:2],  locs2[q,:2], locs2[r,:2]])
        # print('=====')
        H = computeH_norm(l1, l2)
        # print(H)
        # HCV, _ = computeHCV2(l1, l2, cv2.RANSAC)
        # print(HCV)
        # print('=====')

        # pdb.set_trace()

        inliers = [0] * locs1.shape[0]
        for j in range(locs1.shape[0]):
            a_s = locs1[j].T
            b_s = locs2[j].T
            a_ss = H @ a_s
            a_ss = a_ss/a_ss[2]
            sub = b_s - a_ss

            # pdb.set_trace()

            if np.sqrt(np.linalg.norm(sub)) <= inlier_tol:
                inliers[j] = 1
        if sum(inliers) > inliers_best_cnt:
            H_best = H
            inliers_best_cnt = sum(inliers)
            inliers_best = inliers
    # print('H_best')
    # print(H_best)
    # print(inliers_best_cnt)
    # print(inliers_best)
    # print("same: ",same_cnt)

    good1 = []
    good2 = []
    for idx, positive in enumerate(inliers_best):
        if positive == 1:
            good1.append(locs1[idx,:2])
            good2.append(locs2[idx,:2])
    good1 = np.array(good1)
    good2 = np.array(good2)

    H_best = computeH_norm(good1, good2)
    return H_best, inliers_best


def compositeH(H2to1, template, img):
    
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.
    
    # template: porter
    # img: desk

    # TODO: Create mask of same size as template
    mask = np.full((template.shape[0], template.shape[1]), 255, dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # TODO: Warp mask by appropriate homography
    out1 = cv2.warpPerspective(mask, H2to1,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

    # TODO: Warp template by appropriate homography
    out2 = cv2.warpPerspective(template, H2to1,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

    # TODO: Use mask to combine the warped template and the image
    img_sub = cv2.subtract(img, out1)
    img_add = cv2.add(img_sub, out2)    
    return img_add

def computeHCV2(pts_src, pts_dst, method):

    H, status = cv2.findHomography(pts_src, pts_dst, method)
    return [H, status]
