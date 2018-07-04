import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import pandas as pd
import struct
from PIL import Image
#from sklearn import decomposition
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


"""
The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm.
The images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as
to position this point at the center of the 28x28 field.

"""
def read_images(images_name):
#returns an array of flattened images
    f = open(images_name, "rb")
    ds_images = []
    #Let's read the head of the file encoded in 32-bit integers in big-endian(4 bytes)
    mw_32bit = f.read(4)        #magic word
    n_numbers_32bit = f.read(4) #number of images
    n_rows_32bit = f.read(4)    #number of rows of each image
    n_columns_32bit = f.read(4) #number of columns of each image

    #convert it to integers ; '>i' for big endian encoding
    mw =  struct.unpack('>i',mw_32bit)[0]
    n_numbers = struct.unpack('>i',n_numbers_32bit)[0]
    n_rows = struct.unpack('>i',n_rows_32bit)[0]
    n_columns = struct.unpack('>i',n_columns_32bit)[0]

    try:
        for i in range(n_numbers):
            image = []
            for r in range(n_rows):
                for l in range(n_columns):
                    byte = f.read(1)
                    pixel = struct.unpack('B',byte[0])[0]
                    image.append(pixel)
            ds_images.append(image)
    finally:
        f.close()
    return ds_images

def read_labels(labels_name):
    #returns an array of labels
    f = open(labels_name, "rb")
    ds_labels = []
    #Let's read the head of the file encoded in 32-bit integers in big-endian(4 bytes)
    mw_32bit = f.read(4)        #magic word
    n_numbers_32bit = f.read(4) #number of labels

    #convert it to integers ; '>i' for big endian encoding
    mw =  struct.unpack('>i',mw_32bit)[0]
    n_numbers = struct.unpack('>i',n_numbers_32bit)[0]

    try:
        for i in range(n_numbers):
            byte = f.read(1)
            label = struct.unpack('B',byte[0])[0]
            ds_labels.append(label)

    finally:
        f.close()
    return ds_labels

def read_dataset(images_name,labels_name):
    #reads an image-file and a labels file, and returns an array of tuples of (flattened_image, label)
    images = read_images(images_name)
    labels = read_labels(labels_name)
    assert len(images) == len(labels)
    return images,labels

def PCAnalysis(train, test):
    pca = PCA(300, svd_solver='full')
    train = pca.fit_transform(train)
    test = pca.transform(test)
    pca_std = np.std(train)

    #Show variance graph to choose an addequate number of components to keep
    """
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    print("VARIANCE RATIO: " + str(pca.explained_variance_ratio_.cumsum()))
    plt.show()
    #print("NUMBER OF COMPONENTS RETAINED")
    #print(pca.n_components_)
    print("VARIANCE RATIO: " + str(sum(pca.explained_variance_)))
    print(pca.explained_variance_)
    """
    print("PCA SUCCESSFUL")

    return (train, test, pca_std)

def LearnKNN(train, ytrain, test, ytest):
    #.35 accuracy with 3 neighbors
    #.32 accuracy with 4 neighbors
    print("KNN Classifier")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train, ytrain)
    predicted = knn.predict(test)
    acc = accuracy_score(ytest, predicted)
    print(acc)
    print("done")


def LearnKeras(train, ytrain, testset, ytest, pca_std):
    model = Sequential()
    layers = 1
    units = 128
    batch = 60
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)

    model.add(Dense(units, input_dim=300, activation='relu'))
    model.add(GaussianNoise(pca_std))
    for i in range(layers):
        model.add(Dense(units, activation='relu'))
        model.add(GaussianNoise(pca_std))
        model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    #model.fit_generator(generator(train,ytrain,batch_size),
    #epochs=3,steps_per_epoch = train.shape[0]/batch_size, validation_data=generator(testset,ytest,batch_size*2),
    #validation_steps=testset.shape[0]/batch_size*2)
    model.fit(train, ytrain, epochs=100, batch_size=batch, validation_data=(testset,ytest))
    #test_on_batch(self, x, y, sample_weight=None)

testset, ytest = read_dataset("test_images","test_labels") #an array of 70000 labels (0~9)
trainingset, ytrain = read_dataset("train_images","train_labels") #a 70000x784 numpy array which contains all examples with each


trainingset = np.asarray(trainingset)
testset = np.asarray(testset)

"""
fig = plt.figure(figsize=(10,20))
sp = fig.add_subplot(10,5,1)
sp.set_title(8)
plt.axis('off')
image = np.array(trainingset[59999]).reshape(28,28)
plt.imshow(image,interpolation='none',cmap=plb.gray(),label=8)
plt.show()
"""

scaler = StandardScaler()
scaler.fit(trainingset)
trainingset = scaler.transform(trainingset)
testset = scaler.transform(testset)
trainingset, testset, pca_std = PCAnalysis(trainingset, testset)

#LearnKNN(trainingset, ytrain, testset, ytest)
LearnKeras(trainingset, ytrain, testset, ytest, pca_std)
#Accuracy(imageReducedTest, targetTest)
