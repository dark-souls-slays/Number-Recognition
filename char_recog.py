from sklearn.datasets import fetch_mldata
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import pandas as pd
import struct
from PIL import Image
#from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sys

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

def preprocess(data, samples):
    for j in range(samples):
        result = []
        for i in range(784):
            result.append(float(data[j][i])/255.0)
        data[j] = np.array(result)
    print("preprocess successful")
    return data

def PCAnalysis(train, test):
    #784 original image dimension
    """
    pca = PCA(n_components=500)
    pca.fit(train)

    #Show variance to choose an adecuate number of components to keep
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    """
    pca = PCA(100, svd_solver='auto')
    train = pca.fit_transform(train)
    test = pca.transform(test)
    #print("NUMBER OF COMPONENTS RETAINED")
    #print(pca.n_components_)
    print("VARIANCE RATIO: " + str(sum(pca.explained_variance_)))
    print(pca.explained_variance_)

    print("PCA successful")
    
    return (train, test)

def Learn(image, target, test, answer):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(image, target)
    counter = 0
    for i in range(10000):
        if(knn.predict([test[i]]) == answer[i]):
            counter = counter + 1
        else:
            print(knn.predict_proba([test[i]]))
            print(knn.predict([test[i]]))
            print(answer[i])
    print(float(counter)/10000.0)
    return 1

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
trainingset= preprocess(trainingset, 60000)
testset = preprocess(testset, 10000)
trainingset, testset = PCAnalysis(trainingset, testset)
"""
Learn(imageReduced, target, imageReducedTest,targetTest)
#Accuracy(imageReducedTest, targetTest)
"""
