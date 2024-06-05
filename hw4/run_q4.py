import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

test_y = ["TODOLIST\n1MAKEATODOLIST\n2CHECKOFFTHEFIRST\nTHINGONTODOLIST\n"+
          "3REALIZEYOUHAVEALREADYCOMPLETED2THINGS\n4REWARDYOURSELFWITH\nANAP\n"]
test_y.append("ABCDEFG\nHIJKLMN\nOPQRSTU\nVWXYZ\n1234567890\n")
test_y.append("HAIKUSAREEASY\nBUTSOMETIMESTHEYDONTMAKESENSE\nREFRIGERATOR\n")
test_y.append("DEEPLEARNING\nDEEPERLEARNING\nDEEPESTLEARNING\n")


n_img = 0
all_acc = []
for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1, n_img)

    # should not be Greys, should be grey
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    row_num = 0
    cur_maxr = bboxes[0][2]
    bboxes_row_list = []
    bboxes_row_list.append([])
    bboxes.sort(key=lambda box: box[2])
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        # char_image = bw[minr:maxr, minc:maxc]
        # plt.imshow(char_image)
        # plt.show()
        if minr > cur_maxr:
            row_num += 1
            cur_maxr = maxr
            bboxes_row_list.append([])
        bboxes_row_list[row_num].append(bbox)
    print("row_num", row_num+1)
    for row in bboxes_row_list:
        row.sort(key=lambda box: box[3])

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))

    # compute and report test accuracy
    ans_string = ""
    for i, row in enumerate(bboxes_row_list):
        for j, bbox in enumerate(row):
            minr, minc, maxr, maxc = bbox
            char_image = bw[minr:maxr, minc:maxc]
            letter = np.pad(char_image, (9, 9), 'constant',
                            constant_values=1)
            letter_transposed = skimage.transform.resize(letter, (32, 32)).T
            h1 = forward(letter_transposed.reshape(1, 32*32), params, "layer1")
            test_probs = forward(h1, params, "output", softmax)
            character = letters[np.argmax(test_probs)]
            ans_string = ans_string + character
        ans_string = ans_string + "\n"
    print(ans_string)
    acc, string_len = 0, 0
    # print(len(ans_string))
    # print(len(test_y[n_img]))
    for y, y_label in zip(ans_string, test_y[n_img]):
        if y==y_label:
            acc+=1
        string_len += 1
    acc = acc/string_len
    print("acc: ", acc)  
    all_acc.append(acc)      
    n_img+=1
all_acc = [ '%.4f' % elem for elem in all_acc ]
print(all_acc)