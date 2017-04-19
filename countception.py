# An adaptation from https://github.com/ieee8023/NeuralNetwork-Examples/blob/master/theano/counting/count-ception.ipynb


import pickle
from keras.layers import Conv2D, BatchNormalization, Input, concatenate
# from keras.layers import Dense, Activation, Lambda, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
# from keras.datasets import cifar10
import numpy as np
from skimage.io import imread  #, imsave
import scipy.misc
import sys
import glob
import os

scale = 1
patch_size = 32
framesize = 256
noutputs = 1
nsamples = 32
stride = 1

paramfilename = str(scale) + "-" + str(patch_size) + "-cell2_cell_data.p"
datasetfilename = str(scale) + "-" + str(patch_size) + "-" + str(framesize) + "-" + str(stride) + "-cell2-dataset.p"
print(paramfilename)
print(datasetfilename)

imgs = []
for filename in glob.iglob('cells/*cell.png'):
    xml = filename.split("cell.png")[0] + "dots.png"
    imgs.append([filename, xml])


def getMarkersCells(labelPath):
    lab = imread(labelPath)[:, :, 0]/255
    return np.pad(lab, patch_size, "constant")


def getCellCountCells(markers, x_y_h_w, scale):
    x, y, h, w = x_y_h_w
    types = [0] * noutputs
    types[0] = markers[y:y+w, x:x+h].sum()
    return types


def getLabelsCells(img, labelPath, base_x, base_y, stride):
    width =((img.shape[0])//stride)+1
    print("label size: ", width)
    labels = np.zeros((noutputs, width, width))
    markers = getMarkersCells(labelPath)

    for x in range(0, width):
        for y in range(0, width):

            count = getCellCountCells(markers,(base_x + x*stride, base_y + y*stride, patch_size, patch_size), scale) 
            for i in range(0, noutputs):
                labels[i][y][x] = count[i]

    count_total = getCellCountCells(markers,(base_x, base_y, framesize+patch_size, framesize+patch_size), scale)
    return labels, count_total


def getTrainingExampleCells(img_raw, labelPath, base_x, base_y, stride):
    img = img_raw[base_y:base_y+framesize, base_x:base_x+framesize]
    img_pad = np.pad(img, patch_size//2, "constant")
    labels, count = getLabelsCells(img_pad, labelPath, base_x, base_y, stride)
    return img, labels, count


if os.path.isfile(datasetfilename):
    print("reading", datasetfilename)
    dataset = pickle.load(open(datasetfilename, "rb" ))
else:
    dataset = []
    print(len(imgs))
    for path in imgs:

        imgPath = path[0]
        print(imgPath)

        im = imread(imgPath)
        img_raw_raw = im.mean(axis=(2))  # grayscale

        img_raw = scipy.misc.imresize(img_raw_raw,
                                      (img_raw_raw.shape[0]/scale,
                                       img_raw_raw.shape[1]/scale))
        print(img_raw_raw.shape, " ->>>>", img_raw.shape)

        print("input image raw shape", img_raw.shape)

        labelPath = path[1]
        for base_x in range(0, img_raw.shape[0], framesize):
            for base_y in range(0, img_raw.shape[1], framesize):
                img, lab, count = getTrainingExampleCells(img_raw, labelPath, base_y, base_x, stride)
                print("count ", count)
                
                ef = patch_size/stride
                lab_est = [(l.sum()/(ef**2)).astype(np.int) for l in lab]
                print("lab_est", lab_est)
                
                assert count == lab_est
                
                dataset.append((img, lab, count))
                print("img shape", img.shape)
                print("label shape", lab.shape)
                sys.stdout.flush()
                    
    print("writing", datasetfilename)
    out = open(datasetfilename, "wb", 0)
    pickle.dump(dataset, out)
    out.close()
print("DONE")


def SimpleFactory(ch_1x1, ch_3x3, inp):
    conv1x1 = Conv2D(filters=ch_1x1, kernel_size=1,
                     padding='same', activation=LeakyReLU(0.01))(inp)
    conv3x3 = Conv2D(filters=ch_3x3, kernel_size=3,
                     padding='same', activation=LeakyReLU(0.01))(inp)
    return concatenate([conv1x1, conv3x3])


inputs = Input(shape=(64, 64, 3))

bn = BatchNormalization(axis=1, input_shape=(64, 64, 3))(inputs)
c1 = Conv2D(64, (1, 1), padding='same', activation='relu')(bn)
conc = SimpleFactory(16, 16, c1)


print(bn.get_shape())
print(c1.get_shape())
print(conc.get_shape())

# model.add()x2 = conv1x1(x1)
# model.add()y2 = conv3x3(x1)
# model.add()x3 = concatenate(x2, y2)
