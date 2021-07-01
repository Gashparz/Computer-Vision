"""
    PROIECT MOZAIC
"""

# Parametrii algoritmului sunt definiti in clasa Parameters.
from parameters import *
from build_mosaic import *
import pickle



# daca greyscale == 'True' inseamna ca avem de aface cu o poza greyscale
# numele imaginii care va fi transformata in mozaic
image_path = 'C:/Users/eduar/Desktop/Anul 3/Vedere Artificiala/data/imaginiTest/r.jpeg.jpg'

params = Parameters(image_path, False)



# directorul cu imagini folosite pentru realizarea mozaicului
params.small_images_dir = 'C:/Users/eduar/Desktop/Anul 3/Vedere Artificiala/data/colectie/'

# tipul imaginilor din director
params.image_type = 'png'
# numarul de piese ale mozaicului pe orizontala
# pe verticala vor fi calcultate dinamic a.i sa se pastreze raportul
params.num_pieces_horizontal = 100
# afiseaza piesele de mozaic dupa citirea lor
params.show_small_images = False
# modul de aranjarea a pieselor mozaicului
# optiuni: 'aleator', 'caroiaj'
params.layout = 'caroiaj'
# criteriul dupa care se realizeaza mozaicul
# optiuni: 'aleator', 'distantaCuloareMedie', 'DistCulMediaVeciniDif' cea din urma fiind pentru punctul c)
params.criterion = 'DistCulMediaVeciniDif'
# daca params.layout == 'caroiaj', sa se foloseasca piese hexagonale
params.hexagon = False


img_mosaic = build_mosaic(params)
cv.imwrite('mozaic.png', img_mosaic)
