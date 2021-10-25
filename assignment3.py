#from keras.datasets import cifar10
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# Library for plot the output and save to file
import matplotlib as mpl
mpl.use('Agg')

# load data


def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Load the CIFAR10 dataset
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = load_csv(filename)
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Show dimension for each variable
print('Train image shape:    {0}'.format(xTrain.shape))
print('Train label shape:    {0}'.format(yTrain.shape))
print('Validate image shape: {0}'.format(xVal.shape))
print('Validate label shape: {0}'.format(yVal.shape))
print('Test image shape:     {0}'.format(xTest.shape))
print('Test label shape:     {0}'.format(yTest.shape))

# Show some CIFAR10 images
plt.subplot(221)
plt.imshow(xTrain[0])
plt.axis('off')
plt.title(classesName[yTrain[0]])
plt.subplot(222)
plt.imshow(xTrain[1])
plt.axis('off')
plt.title(classesName[yTrain[1]])
plt.subplot(223)
plt.imshow(xVal[0])
plt.axis('off')
plt.title(classesName[yVal[1]])
plt.subplot(224)
plt.imshow(xTest[0])
plt.axis('off')
plt.title(classesName[yTest[0]])
plt.clf()

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Add bias dimension columns
xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])


class Svm (object):
    """" Svm classifier """

    def __init__(self, inputDim, outputDim):
        self.W = None

        sigma = 0.01
        self.W = sigma * np.random.randn(inputDim, outputDim)

    def calLoss(self, x, y, reg):

        loss = 0.0
        dW = np.zeros_like(self.W)

        s = x.dot(self.W)
        # Score with yi
        s_yi = s[np.arange(x.shape[0]), y]
        # finding the delta
        delta = s - s_yi[:, np.newaxis]+1
        # loss for samples
        loss_i = np.maximum(0, delta)
        loss_i[np.arange(x.shape[0]), y] = 0
        loss = np.sum(loss_i)/x.shape[0]
        # Loss with regularization
        loss += reg*np.sum(self.W*self.W)
        # Calculating ds
        ds = np.zeros_like(delta)
        ds[delta > 0] = 1
        ds[np.arange(x.shape[0]), y] = 0
        ds[np.arange(x.shape[0]), y] = -np.sum(ds, axis=1)

        dW = (1/x.shape[0]) * (x.T).dot(ds)
        dW = dW + (2 * reg * self.W)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train(self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None

            num_train = np.random.choice(x.shape[0], batchSize)
            xBatch = x[num_train]
            yBatch = y[num_train]
            loss, dW = self.calLoss(xBatch, yBatch, reg)
            self.W = self.W - lr * dW
            lossHistory.append(loss)

            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict(self, x,):

        yPred = np.zeros(x.shape[0])

        s = x.dot(self.W)
        yPred = np.argmax(s, axis=1)

        return yPred

    def calAccuracy(self, x, y):
        acc = 0

        yPred = self.predict(x)
        acc = np.mean(y == yPred)*100

        return acc


numClasses = np.max(yTrain) + 1

print('Start training Svm classifier')

classifier = Svm(xTrain.shape[1], numClasses)

# Show weight for each class before training
if classifier.W is not None:
    tmpW = classifier.W[:-1, :]
    tmpW = tmpW.reshape(32, 32, 3, 10)
    tmpWMin, tmpWMax = np.min(tmpW), np.max(tmpW)
    for i in range(numClasses):
        plt.subplot(2, 5, i+1)
        plt.title(classesName[i])
        wPlot = 255.0 * (tmpW[:, :, :, i].squeeze() -
                         tmpWMin) / (tmpWMax - tmpWMin)
        plt.imshow(wPlot.astype('uint8'))
    plt.clf()

# Training classifier
startTime = time.time()
classifier.train(xTrain, yTrain, lr=1e-7, reg=5e4, iter=1500, verbose=True)
print('Training time: {0}'.format(time.time() - startTime))
print('Training acc:   {0}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print('Validating acc: {0}%'.format(classifier.calAccuracy(xVal, yVal)))
print('Testing acc:    {0}%'.format(classifier.calAccuracy(xTest, yTest)))
