import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

def draw_rect(rect, c, w):
    # rect = [int(x) for x in rect]
    rect = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], 
                             linewidth=w, edgecolor=c, facecolor='none')
    return rect

def update_fig(i):
    ax.clear()
    ax.imshow(seq[:, :, i], cmap='gray')
    ax.set_title('frame ' + str(i))
    ax.axis('off')
    ax.add_patch(draw_rect(girlseqrects[i], 'r', 2))
    ax.add_patch(draw_rect(girlseqrects_no[i],'b', 1))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    # default=1,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

girlseqrects_no = [0] * seq.shape[2]
girlseqrects_no[0] = rect
girlseqrects = [0] * seq.shape[2]
girlseqrects[0] = rect

initial_image = seq[:,:,0]
prev_p = np.zeros(2)
template = seq[:,:,0]
for t in range(1, seq.shape[2]):
    # print('========time: ' + str(t))
    image1 = seq[:,:,t]
    rect_prev = girlseqrects[t-1]
    
    p = LucasKanade(template, image1, rect_prev, threshold, num_iters,p0=prev_p)
    # we need to account fot the displacement from first rect to current rect
    p_accu = [rect_prev[0] - rect[0], rect_prev[1]-rect[1]]
    p_star = LucasKanade(initial_image, image1, girlseqrects[0], threshold, num_iters, p0=(p+p_accu))

    if np.linalg.norm((p+p_accu)-p_star) <= template_threshold:
        print("smaller")
        # we need to account fot the displacement from first rect to current rect
        p_diff = p_star - p_accu
        rect1 = [rect_prev[0]+p_diff[0], rect_prev[1]+p_diff[1], rect_prev[2]+p_diff[0], rect_prev[3]+p_diff[1]]
        girlseqrects[t] = rect1
        template = image1
        prev_p = np.zeros(2)
    else:
        print("larger")
        # there must be a problem and so we act conservatively by not updating the template in this step.
        # rect1 = [rect_prev[0]+p[0], rect_prev[1]+p[1], rect_prev[2]+p[0], rect_prev[3]+p[1]]
        # carseqrects[t] = rect1
        girlseqrects[t] = rect_prev
        prev_p = p
        # don't update template

    # vanilla lucaskanade
    image0 = seq[:,:,t-1] 
    p_ori = LucasKanade(image0, image1, girlseqrects_no[t-1], threshold, num_iters)
    rect0 = girlseqrects_no[t-1]
    rect1 = [rect0[0]+p_ori[0], rect0[1]+p_ori[1], rect0[2]+p_ori[0], rect0[3]+p_ori[1]]
    girlseqrects_no[t] = rect1

np.save('../data/girlseqrects-wcrt.npy', girlseqrects)

ct = [0, 1, 20, 40, 60, 80]
fig, axs = plt.subplots(2, 3, figsize=(20, 20))
for i, ax in enumerate(axs.reshape(-1)):
    t = ct[i]
    ax.imshow(seq[:,:,t], cmap='gray')
    ax.set_title('frame ' + str(t))
    ax.axis('off')
    ax.add_patch(draw_rect(girlseqrects[t],'r', 2))
    ax.add_patch(draw_rect(girlseqrects_no[t],'b', 1))
plt.savefig('../data/girlseqrects-wcrt.png', bbox_inches='tight')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 20))
# Create the animation
ani = animation.FuncAnimation(fig, update_fig, frames=seq.shape[2], interval=200)  # Adjust interval as needed
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Li-Wei'), bitrate=1800)
ani.save('../data/girlseqresult-wcrt.mp4', writer=writer)