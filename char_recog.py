from sklearn.datasets import fetch_mldata
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import struct
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
    return zip(images,labels)

def preprocess(image1D):
    image1DNew =  np.array([])
    image1DNew.astype(float)
    for i in range(784):
        result = float(image1D[i])/255.0
        image1DNew = np.append(image1DNew, result)
    return image1DNew

def PCA(image1D):
    #784 original image dimension
    """
    image = image1D.reshape(28,28)
    Rx = np.corrcoef(image)
    print(np.shape(Rx))
    print(Rx)
    w, v = LA.eig(np.diag((Rx)))
    print("Eigenvalues Rx")
    print(w)

    #Find proper dimensionality
    while (newD<11):
        newD = newD + 1
    """
    sys.setrecursionlimit(10000)
    pca = PCA(52)
    pca.fit(image1D)
    print(np.shape(pca.components_))
        #ratio = sum(pca.explained_variance_) / sum(w)
    """
    # access values and vectors
    print(pca.components_)
    print(pca.explained_variance_)
    # transform data
    B = pca.transform(A)
    print("Projection")
    print(B)
    """

#testset = read_dataset("test_images","test_labels") #an array of 70000 labels (0~9)
trainingset = read_dataset("train_images","train_labels") #a 70000x784 numpy array which contains all examples with each
dataset = trainingset

fig = plt.figure(figsize=(10,20))

for i in range(1):
    sp = fig.add_subplot(10,5,i+1)
    sp.set_title(dataset[i][1])
    plt.axis('off')
    image1D = np.array(dataset[i][0])#.reshape(28,28)#first 50 images
    print("IMAGE IN 1D")
    print(image1D)
    image1DNew = preprocess(image1D)
    print("IMAGE AFTER PREPROCESSING")
    print(image1DNew)
    #REDUCING DIMENSION
    PCA(image1DNew)
    image = image1D.reshape(28,28)#first 50 images
    #print(image)
    #plt.imshow(image,interpolation='none',cmap=plb.gray(),label=dataset[i][1])
#plt.show()
