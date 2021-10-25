from tensorflow.keras.datasets.cifar10 import load_data
from keras.callbacks import EarlyStopping
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# N = 5
# menMeans = (20, 35, 30, 35, -27)
# womenMeans = (25, 32, 34, 20, -25)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
# ind = np.arange(N)    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence

# fig, ax = plt.subplots()

# p1 = ax.bar(ind, menMeans, width, yerr=menStd, label='Men')
# p2 = ax.bar(ind, womenMeans, width,
#             bottom=menMeans, yerr=womenStd, label='Women')

# ax.axhline(0, color='grey', linewidth=0.8)
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(ind)
# ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
# ax.legend()

# # Label with label_type 'center' instead of the default 'edge'
# ax.bar_label(p1, label_type='center')
# ax.bar_label(p2, label_type='center')
# ax.bar_label(p2)

# plt.show()

# fig, ax = plt.subplots()  # Create a figure containing a single axes.
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.

x = np.linspace(0, 2, 100)

# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x, x, label='linear')  # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
ax.plot(x, x**3, label='cubic')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend           .
# inserting necessary libraries


# loading the ciphar dataset

# To load the MNIST digit dataset
(X_train, Y_train), (X_text, Y_test) = load_data()  # loading data
# chking the training dataset
print("There  are", len(X_train), "images in the training dataset")
# checking total test dataset images
print("There are", len(X_text), "images in the test dataset")
X_train[0].shape


# code to view images
num_rows, num_cols = 2, 5

f, ax = plt.subplots(num_rows, num_cols, figsize=(12, 5), gridspec_kw={
                     'wspace': 0.03, 'hspace': 0.01}, squeeze=True)

for r in range(num_rows):
    for c in range(num_cols):
        image_index = r * 5 + c
        ax[r, c].imshow(X_train[image_index], cmap='gray')
        ax[r, c].set_title('No.%d' % Y_train[image_index])
plt.show()
plt.close()
