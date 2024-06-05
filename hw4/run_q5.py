import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

np.random.seed(777)

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024, 32, params, "layer1")
initialize_weights(32, 32, params, "layer2")
initialize_weights(32, 32, params, "layer3")
initialize_weights(32, 1024, params, "output")

params["m"+"Wlayer1"] = np.zeros((1024, 32))
params["m"+"blayer1"] = np.zeros(32)
params["m"+"Wlayer2"] = np.zeros((32, 32))
params["m"+"blayer2"] = np.zeros(32)
params["m"+"Wlayer3"] = np.zeros((32, 32))
params["m"+"blayer3"] = np.zeros(32)
params["m"+"Woutput"] = np.zeros((32, 1024))
params["m"+"boutput"] = np.zeros(1024)

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   use 'm_'+name variables in initialize_weights from nn.py
        #   to keep a saved value
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward pass
        h1 = forward(xb, params, "layer1", relu)
        h2 = forward(h1, params, "layer2", relu)
        h3 = forward(h2, params, "layer3", relu)
        out = forward(h3, params, "output", sigmoid)

        # loss
        loss = np.sum((xb - out)**2)
        total_loss += loss
        
        # backward
        delta1 = 2*(out-xb)
        delta2 = backwards(delta1, params, "output", sigmoid_deriv)
        delta3 = backwards(delta2, params, 'layer3', relu_deriv)
        delta4 = backwards(delta3, params, 'layer2', relu_deriv)
        delta5 = backwards(delta4, params, 'layer1', relu_deriv)


        # apply gradient, remember to update momentum as well
        # v = mu * v - learning_rate * dx # integrate velocity
        # x += v # integrate position
        mu = 0.9     
        params["m"+"Wlayer1"] = mu*params["m"+"Wlayer1"] - learning_rate*params["grad_W" + 'layer1']
        params["m"+"blayer1"] = mu*params["m"+"blayer1"] - learning_rate*params["grad_b" + 'layer1']
        params["m"+"Wlayer2"] = mu*params["m"+"Wlayer2"] - learning_rate*params["grad_W" + 'layer2']
        params["m"+"blayer2"] = mu*params["m"+"blayer2"] - learning_rate*params["grad_b" + 'layer2']
        params["m"+"Wlayer3"] = mu*params["m"+"Wlayer3"] - learning_rate*params["grad_W" + 'layer3']
        params["m"+"blayer3"] = mu*params["m"+"blayer3"] - learning_rate*params["grad_b" + 'layer3']
        params["m"+"Woutput"] = mu*params["m"+"Woutput"] - learning_rate*params["grad_W" + 'output']
        params["m"+"boutput"] = mu*params["m"+"boutput"] - learning_rate*params["grad_b" + 'output']

        params["Wlayer1"] += params["m"+"Wlayer1"]
        params["blayer1"] += params["m"+"blayer1"]
        params["Wlayer2"] += params["m"+"Wlayer2"]
        params["blayer2"] += params["m"+"blayer2"]
        params["Wlayer3"] += params["m"+"Wlayer3"]
        params["blayer3"] += params["m"+"blayer3"]
        params["Woutput"] += params["m"+"Woutput"]
        params["boutput"] += params["m"+"boutput"]        
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
print(visualize_x.shape)
reconstructed_x = np.zeros(visualize_x.shape)
for i in range(visualize_x.shape[0]):
    xb = visualize_x[i, :]
    h1 = forward(xb, params, "layer1", relu)
    h2 = forward(h1, params, "layer2", relu)
    h3 = forward(h2, params, "layer3", relu)
    reconstructed = forward(h3, params, "output", sigmoid)
    reconstructed_x[i] = reconstructed

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
psnr = []
for valid in valid_x:
    h1 = forward(valid, params, "layer1", relu)
    h2 = forward(h1, params, "layer2", relu)
    h3 = forward(h2, params, "layer3", relu)
    out_x = forward(h3, params, "output", sigmoid)
    psnr.append(peak_signal_noise_ratio(valid, out_x))
print("average psnr: ", sum(psnr)/len(psnr))