import sys
import cv2 as cv
import numpy as np
import copy

from parameters import *
from select_path import *

import pdb


def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea pentru fiecare pixel al imaginii

    E = np.zeros((img.shape[0], img.shape[1]))

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1)

    E = np.abs(sobelx) + np.abs(sobely)

    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:, :, 0] = E.copy()
    new_image_E[:, :, 1] = E.copy()
    new_image_E[:, :, 2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path):
    """
    elimina drumul vertical din imagine
    :param img: imaginea initiala
    :path - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = path[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, col + 1:].copy()

    return updated_img


def decrease_width(params: Parameters, num_pixels):
    img = params.image.copy()  # copiaza imaginea originala

    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol                
        E = compute_energy(img)
        # print(E.shape)
        path = select_path(params, E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img


def decrease_height(params: Parameters, num_pixels):
    img = params.image.copy()
    # folosim tot functia decrease_width doar ca rotim imaginea cu 90 de grade la dreapta

    img = cv.rotate(img, cv.cv2.ROTATE_90_CLOCKWISE)

    for i in range(num_pixels):
        print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        E = compute_energy(img)

        path = select_path(params, E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    # revenim la forma initiala
    img = cv.rotate(img, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv.destroyAllWindows()
    return img


def amplify_content(params: Parameters):
    img = params.image.copy()

    # calculam numaul de pixeli necesari pentru amplificare
    width = int(img.shape[1] * params.factor_amplification)
    height = int(img.shape[0] * params.factor_amplification)

    # scalam imaginea
    img_upscaled = cv.resize(img, (width, height))

    # calculam diferenta de pixeli
    num_pixel_width = img_upscaled.shape[1] - img.shape[1]
    num_pixel_height = img_upscaled.shape[0] - img.shape[0]

    # am rescris functi de decrease_height si decrease_width
    for i in range(num_pixel_width):
        print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixel_width))

        E = compute_energy(img_upscaled)

        path = select_path(params, E, params.method_select_path)
        if params.show_path:
            show_path(img_upscaled, path, params.color_path)
        img_upscaled = delete_path(img_upscaled, path)

    img_rot = cv.rotate(img_upscaled, cv.cv2.ROTATE_90_CLOCKWISE)

    for i in range(num_pixel_height):
        print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixel_height))

        E = compute_energy(img_rot)

        path = select_path(params, E, params.method_select_path)
        if params.show_path:
            show_path(img_rot, path, params.color_path)
        img_rot = delete_path(img_rot, path)

    img = cv.rotate(img_rot, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv.destroyAllWindows()
    return img


def delete_object(params: Parameters):
    img = params.image.copy()


    for i in range(params.h):
        print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, params.h))

        E = compute_energy(img)

        path = select_path(params, E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    return img


def resize_image(params: Parameters):
    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image


    elif params.resize_option == 'micsoreazaInaltime':
        resized_image = decrease_height(params, params.num_pixel_height)
        return resized_image

    elif params.resize_option == 'amplificaContinut':
        resized_image = amplify_content(params)
        return resized_image

    elif params.resize_option == 'eliminaObiect':
        # transformam imaginea in uint8 deoarece nu mergea foarte ok inainte selectROI
        img = np.uint8(params.image)
        # salvam return-ul functiei in parameters
        params.y0, params.x0, params.h, params.w = cv.selectROI(img)

        deletedObject = delete_object(params)
        return deletedObject


    else:
        print('The option is not valid!')
        sys.exit(-1)
