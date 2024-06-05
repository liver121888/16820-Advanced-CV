import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from LucasKanade import LucasKanade

def draw_rect(rect, c):
    # rect = [int(x) for x in rect]
    rect = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], 
                             linewidth=1, edgecolor=c, facecolor='none')
    return rect

def update_fig(i):
    ax.clear()
    ax.imshow(seq[:, :, i], cmap='gray')
    ax.set_title('frame ' + str(i))
    ax.axis('off')
    ax.add_patch(draw_rect(girlseqrects[i], 'r'))

# write your script here, we recommend the above libraries for making your animation
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
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

# seq = (255*seq).astype(np.uint8)
# x1 y1 x2 y2
# print('start_rect: ' + str(rect))
# print(seq.shape)
# h,w,t = (240, 320, 415)
girlseqrects = [0] * seq.shape[2]
girlseqrects[0] = rect
step = 1
for t in range(step, seq.shape[2], step):
    # print('========time: ' + str(t))
    image0 = seq[:,:,t-step]
    image1 = seq[:,:,t]
    p = LucasKanade(image0, image1, girlseqrects[t-step], threshold, num_iters)
    # print(p)
    rect0 = girlseqrects[t-step]
    rect1 = [rect0[0]+p[0], rect0[1]+p[1], rect0[2]+p[0], rect0[3]+p[1]]
    # print('rect1:' + str(rect1))
    girlseqrects[t] = rect1

np.save('../data/girlseqrects.npy', girlseqrects)

ct = [0, 1, 20, 40, 60, 80]
fig, axs = plt.subplots(2, 3, figsize=(20, 20))
for i, ax in enumerate(axs.reshape(-1)):
    t = ct[i]
    ax.imshow(seq[:,:,t], cmap='gray')
    ax.set_title('frame ' + str(t))
    ax.axis('off')
    ax.add_patch(draw_rect(girlseqrects[t],'r'))
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('../data/girlseqresult.png', bbox_inches='tight')



# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 20))
# Create the animation
ani = animation.FuncAnimation(fig, update_fig, frames=seq.shape[2], interval=200)  # Adjust interval as needed
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Li-Wei'), bitrate=1800)
ani.save('../data/girlseqresultanimation.mp4', writer=writer)