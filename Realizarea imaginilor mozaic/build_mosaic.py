import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb

from add_pieces_mosaic import *
from parameters import *
import pickle

#o functie facuta pentru a extrage setul cifar
def unbach(file):
    with open(file, mode='rb') as fin:
        batch = pickle.load(fin, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

# citim piesele pickled si le stocam in images
def load_pieces_pickle(params: Parameters):
    images = []


    images_batch, labels = unbach(params.batch_dir)
    for i in range(len(labels)):
        if labels[i] == 4:
            images.append(images_batch[i])
    images = np.array(images, np.float32)

    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i
    images_col = []
    images_gray = []
    images_dir = os.listdir(params.small_images_dir)
    for images_name in images_dir:
        img_current = cv.imread(params.small_images_dir + images_name)
        images_col.append(img_current)
        images_gray.append(cv.cvtColor(img_current, cv.COLOR_BGR2GRAY))

    if params.grayscale:
        images = np.array(images_gray, np.float32)
    else:
        images = np.array(images_col, np.float32)

    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala

    if params.grayscale:
        H, W = params.image.shape
    else:
        H, W, C = params.image.shape

    # salvam aspect ratio-ul imaginii
    aspect_ratio = H / W

    # redimensioneaza imaginea numarul pieselor pe orizontala inmultit cu latimea
    new_w = params.num_pieces_horizontal * params.small_images.shape[2]
    # calculam o noua inaltime momentana a imaginii
    new_h_ = new_w * aspect_ratio
    # calculam numarul de piese pe verticala
    params.num_pieces_vertical = int(np.ceil(new_h_ / params.small_images.shape[1]))
    new_h = params.num_pieces_vertical * params.small_images.shape[1]
    # dam resize imaginii dupa noile dimensiuni
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)
    # calculam culoara medie atunci cand incarcam piesele
    params.mean_color_pieces = np.float32(np.mean(params.small_images, axis=(1, 2)))
    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
