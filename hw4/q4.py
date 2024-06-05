from  skimage.measure import label, regionprops
from  skimage.color import rgb2gray
import skimage.restoration
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import closing, square, dilation, erosion
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import matplotlib


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image, n_img):
    bboxes = []
    bw = None

    if n_img == 3:
        d = 25
        ext = 20
    else:
        d = 5   
        ext = 5

    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    # image = gaussian(image, sigma=1.5)
    # image = skimage.restoration.denoise_wavelet(image, sigma=2)
    image = rgb2gray(image)
    thresh = threshold_otsu(image)
    bw = closing(image < thresh, square(12))
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(bw)
    # plt.show()
    bw = dilation(bw, footprint=square(d))  

    cleared = clear_border(bw)
    labels = label(cleared)

    for props in regionprops(labels):
        if props.area >= 500:
            minr, minc, maxr, maxc = props.bbox
            ext_bbox = [minr-ext, minc-ext, maxr+ext, maxc+ext]
            bboxes.append(ext_bbox)
    # turn foreground to black and background to white
    bw = 1.0 - bw  
    return bboxes, bw
