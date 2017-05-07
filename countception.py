# An adaptation from https://github.com/ieee8023/NeuralNetwork-Examples/blob/master/theano/counting/count-ception.ipynb
import pickle
from keras.layers import Conv2D, BatchNormalization, Input, concatenate, ZeroPadding2D
# from keras.layers import Dense, Activation, Lambda, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
import keras.models
# from keras.datasets import cifar10
import numpy as np
from skimage.io import imread  #, imsave
import scipy.misc
import sys
import glob
import os
import random


scale = 1
patch_size = 32
framesize = 256
noutputs = 1
nsamples = 32
stride = 1
ef = (patch_size/stride)**2

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


np_dataset = np.asarray(dataset)

random.shuffle(np_dataset)

# TODO: KANSKE SKA CASTAS TILL FLOAT !!!!!!
np_dataset = np.rollaxis(np_dataset,1,0)
np_dataset_x = np.asarray([np.transpose(np.asarray([n]), (1,2,0)) for n in np_dataset[0]])
np_dataset_y = np.asarray([np.transpose(n, (1,2,0)) for n in np_dataset[1]])
np_dataset_c = np.asarray([n for n in np_dataset[2]])

print("np_dataset_x", np_dataset_x.shape)
print("np_dataset_y", np_dataset_y.shape)
print("np_dataset_c", np_dataset_c.shape)

del np_dataset

length = len(np_dataset_x)

n = nsamples

np_dataset_x_train = np_dataset_x[0:n]
np_dataset_y_train = np_dataset_y[0:n]
np_dataset_c_train = np_dataset_c[0:n]
print("np_dataset_x_train", len(np_dataset_x_train))

np_dataset_x_valid = np_dataset_x[n:2*n]
np_dataset_y_valid = np_dataset_y[n:2*n]
np_dataset_c_valid = np_dataset_c[n:2*n]
print("np_dataset_x_valid", len(np_dataset_x_valid))

np_dataset_x_test = np_dataset_x[100:]
np_dataset_y_test = np_dataset_y[100:]
np_dataset_c_test = np_dataset_c[100:]
print("np_dataset_x_test", len(np_dataset_x_test))
print(np_dataset_x_train [:4,0].shape)

# Keras stuff

def ConvFactory(filters, kernel_size, padding, inp, name, padding_type='valid'):
    if padding != 0:
        padded = ZeroPadding2D(padding)(inp)
    else:
        padded = inp
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding_type, name=name+"_conv")(padded)
    activated = LeakyReLU(0.01)(conv)
    bn = BatchNormalization(name=name+"_bn")(activated)
    return bn

def SimpleFactory(ch_1x1, ch_3x3, inp, name):
    conv1x1 = ConvFactory(ch_1x1, 1, 0, inp, name + "_1x1")
    conv3x3 = ConvFactory(ch_3x3, 3, 1, inp, name + "_3x3")
    return concatenate([conv1x1, conv3x3])


def build_model():
    print('#'*80)
    print('# Building model...')
    print('#'*80)

    inputs = Input(shape=(256, 256, 1))
    print("inputs:", inputs.shape)
    c1 = ConvFactory(64, 3, patch_size, inputs, "c1")
    print("c1", c1.shape)
    net1 = SimpleFactory(16, 16, c1, "net1")
    print("net:", net1.shape)
    net2 = SimpleFactory(16, 32, net1, "net2")
    print("net:", net2.shape)
    net3 = ConvFactory(16, 14, 0, net2, "net3")
    print("net:", net3.shape)
    net4 = SimpleFactory(112, 48, net3, "net4")
    print("net:", net4.shape)
    net5 = SimpleFactory(64, 32, net4, "net5")
    print("net:", net5.shape)
    net6 = SimpleFactory(40, 40, net5, "net6")
    print("net:", net6.shape)
    net7 = SimpleFactory(32, 96, net6, "net7")
    print("net:", net7.shape)
    net8 = ConvFactory(32, 17, 0, net7, "net8")
    print("net:", net8.shape)
    net9 = ConvFactory(64, 1, 0, net8, "net9")
    print("net:", net9.shape)
    net10 = ConvFactory(64, 1, 0, net9, "net10")
    print("net:", net10.shape)
    # net11 = ConvFactory(1, 1, 0, net10, "net11")
    final = Conv2D(1, 1, name="final")(net10)
    print("net:", final.shape)


    model = keras.models.Model(inputs=inputs, outputs=final)
    print("Model params:", model.count_params())

    model.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'], learning_rate = 0.005)

    return model

def sum_count_map(m, ef=ef):
    return np.asarray([np.sum(p)/ef for p in m])

TRAIN=True

if TRAIN:
    batch_size = 2
    epochs = 1

    model = build_model()
    hist = model.fit(np_dataset_x_train, np_dataset_y_train, epochs=epochs, batch_size = batch_size,
                    validation_data = (np_dataset_x_valid, np_dataset_y_valid))

    model.save_weights('model.h5')

else:
    model = build_model()
    model.load_weights("model.h5", by_name=True)

pred = model.predict(np_dataset_x_test, batch_size=1)
preds = sum_count_map(pred)
madetests = sum_count_map(np_dataset_y_test)
tests = np.concatenate(np_dataset_c_test)
order = np.argsort(tests)
print(preds[order])
print(madetests[order])
print(tests[order])
