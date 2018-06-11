from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import struct



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
    #reads an image-file and a labels file, and returns an array of tuples of
    #(flattened_image, label)
    images = read_images(images_name)
    labels = read_labels(labels_name)
    assert len(images) == len(labels)
    return zip(images,labels)


testset = read_dataset("test_images","test_labels")
trainingset = read_dataset("train_images","train_labels")
dataset = trainingset

fig = plt.figure(figsize=(10,20))
for i in range(50):
    sp = fig.add_subplot(10,5,i+1)
    sp.set_title(dataset[i][1])
    plt.axis('off')
    image = np.array(dataset[i][0]).reshape(28,28)
    plt.imshow(image,interpolation='none',cmap=plb.gray(),label=dataset[i][1])

plt.show()
