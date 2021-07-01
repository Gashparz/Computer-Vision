import cv2 as cv


# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
import numpy as np


class Parameters:

    def __init__(self, image_path, grayscale):
        self.image_path = image_path
        self.grayscale = grayscale
        # un flag pentru grayscale
        if self.grayscale:
            img = cv.imread(image_path)
            self.image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            self.image = cv.imread(image_path)
        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)
        self.image = np.float32(self.image)
        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        # pentru a extrage noile small images
        self.batch_dir = 'C:/Users/eduar/Desktop/Anul 3/Vedere Artificiala/Tema1 Rezolvare/cifar-10-batches-py/data_batch_1'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None
        self.mean_color_pieces = None
