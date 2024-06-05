import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend all or some of the above libraries for making your animation
def update_fig(i):
    ax.clear()
    ax.imshow(seq_masked[:,:,:,i])
    ax.set_title('frame ' + str(i))
    ax.axis('off')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=20,
    # default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq',
    default='../data/aerialseq.npy',
)

args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq

seq = np.load(seq_file)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''
seq = (255*seq).astype(np.uint8)
image = seq[:,:,0]
image1 = seq[:,:,1]
h, w = image.shape[0], image.shape[1]

mask = np.zeros(image1.shape, dtype=bool)
masks = [0] * seq.shape[2]
masks[0] = mask
step = 1
tfinal = seq.shape[2]
# tfinal = 2
for t in range(step, tfinal, step):
    # print('========time: ' + str(t))
    image0 = seq[:,:,t-step]
    image1 = seq[:,:,t]
    # M = LucasKanadeAffine(image0, image1, threshold, num_iters)
    # print(M)
    mask = SubtractDominantMotion(image0, image1, threshold, num_iters, tolerance)
    # mask = SDM(image0, image1, num_iters, threshold, tolerance)
    masks[t] = mask

seq2 = seq.copy()
for i in range(len(masks)):
    seq2[:,:,i][masks[i]] = 255
seq_masked = np.stack([seq, seq, seq2],axis=-2)
print(seq_masked.shape)
# pdb.set_trace()

ct = [0, 30, 60, 90, 120]
# ct = [0, 1]
fig, axs = plt.subplots(2, 3, figsize=(20, 20))
axs = axs.reshape(-1)
for i in range(len(ct)):
    t = ct[i]
    ax = axs[i]
    ax.imshow(seq_masked[:,:,:,t])
    ax.set_title('frame ' + str(t))
    ax.axis('off')

plt.savefig(seq_file.replace('.npy','') + '.png', bbox_inches='tight')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 20))
# Create the animation
ani = animation.FuncAnimation(fig, update_fig, frames=seq.shape[2], interval=200)  # Adjust interval as needed
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Li-Wei'), bitrate=1800)
ani.save(seq_file.replace('.npy','') + '.mp4', writer=writer)