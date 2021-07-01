from parameters import *
import numpy as np
import pdb
import timeit


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.grayscale:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    else:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape

    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                if params.grayscale:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                else:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                # extragem patch ul

                if params.grayscale:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W]
                    # calculam culoarea sa medie
                    patch_mean_color = np.mean(patch)
                    # caulculam distantele culoriilor medii
                    sorted_indexes = get_sorted_indexes(params.mean_color_pieces, patch_mean_color, params)
                    # primul index este cel mai apropiat de patch ul nostru
                    index = sorted_indexes[0]
                    # facem inlocuirea
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                else:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :]
                    # calculam culoarea sa medie
                    patch_mean_color = np.mean(patch, axis=(0, 1))
                    # caulculam distantele euclidiene ale culoriilor medii
                    sorted_indexes = get_sorted_indexes(params.mean_color_pieces, patch_mean_color, params)
                    # primul index este cel mai apropiat de patch ul nostru
                    index = sorted_indexes[0]
                    # facem inlocuirea
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
    elif params.criterion == "DistCulMediaVeciniDif":

        # folosim o matrice de vecini initializata cu -1
        neigh = np.zeros((params.num_pieces_vertical, params.num_pieces_horizontal)) - 1

        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                # extragem patch ul si calculam culoarea sa medie
                if params.grayscale:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W]
                    patch_mean_color = np.mean(patch)
                else:
                    patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :]
                    patch_mean_color = np.mean(patch, axis=(0, 1))

                sorted_indices = get_sorted_indexes(params.mean_color_pieces, patch_mean_color, params)
                index = sorted_indices[0]

                # incepem de pe prima linie si verificam vecinii in dreapta
                if i == 0 and j > 0:
                    if neigh[i][j - 1] == index:
                        index = sorted_indices[1]

                # suntem pe prima coloana si verificam vecinii de jos
                if i > 0 and j == 0:
                    if neigh[i - 1][j] == index:
                        index = sorted_indices[1]

                # verificam restul in sus si in stanga sa fie diferit
                if i > 0 and j > 0:
                    aux = 0
                    while neigh[i - 1][j] == sorted_indices[aux] or neigh[i][j - 1] == sorted_indices[aux]:
                        aux += 1
                    index = sorted_indices[aux]

                # punem indexul in vectorul de vecini pentru a stii sa nu il folosim langa vecini
                neigh[i, j] = index
                # inlocuim patch ul
                if params.grayscale:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                else:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]


    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def get_sorted_indexes(mean_color_pieces, patch_mean_color, params: Parameters):
    # calculam distantele euclidiene pentur a afla cea mai apropiata floare de patch ul nostru
    if params.grayscale:
        distances = ((mean_color_pieces - patch_mean_color) ** 2)
    else:
        distances = np.sum(((mean_color_pieces - patch_mean_color) ** 2), axis=1)
    # intoarcem array ul sortat
    return distances.argsort()


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    if params.grayscale:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape

        # initializam cu zero
        img_mosaic = np.zeros((h + H, w + W), np.uint8)

        # construim o noua imagine cu un padding in plus in cazul in care random se va pune pe margine
        # si va iesi din poza
        bigger_image = np.zeros((h + (H - 1), w + (W - 1)), np.uint8)
        # copiem imaginea
        bigger_image[:-(H - 1), :-(W - 1)] = params.image_resized
    else:

        # la fel ca mai sus doar ca pentru poze color
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
        img_mosaic = np.zeros((h + H, w + W, c), np.uint8)

        bigger_image = np.zeros((h + (H - 1), w + (W - 1), c), np.uint8)
        bigger_image[:-(H - 1), :-(W - 1), :] = params.image_resized

    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    free_pixels = np.zeros((img_mosaic.shape[0], img_mosaic.shape[1]), np.int)
    for i in range(free_pixels.shape[0]):
        for j in range(free_pixels.shape[1]):
            # o initilizam cu numarul fiecarui pixel din imagine (ex pixelul 1 va avea valoarea 1)
            free_pixels[i, j] = i * free_pixels.shape[1] + j

    # paddingul este inlocuit cu -1 pentru a nu putea fi selectat de algoritm
    free_pixels[h:, :] = -1
    free_pixels[:, w:] = -1
    while True:
        # extragem pixelii nefolositi ( liberi )
        free_ = free_pixels[free_pixels > -1]
        # cand nu mai sunt pixeli liberi algoritmul se opreste
        if len(free_) == 0:
            break
        # luam random cate un punct din matrice, nu se poate pune problema ca punctul sa fie ocupat deja
        index_free = np.random.randint(low=0, high=len(free_), size=1)
        # salvam pozitia cestuia
        row = int(free_[index_free] / free_pixels.shape[1])
        col = int(free_[index_free] % free_pixels.shape[1])

        if params.grayscale:
            # extragem patch ul
            patch = bigger_image[row: row + H, col: col + W]
            # calculam media culorii patch ului
            patch_mean_color = np.mean(patch)

            index = get_sorted_indexes(params.mean_color_pieces, patch_mean_color, params)[0]

            img_mosaic[row:row + H, col:col + W] = params.small_images[index]

        else:
            patch = bigger_image[row: row + H, col: col + W, :]
            patch_mean_color = np.mean(patch, axis=(0, 1))

            index = get_sorted_indexes(params.mean_color_pieces, patch_mean_color, params)[0]

            img_mosaic[row:row + H, col:col + W, :] = params.small_images[index]

        # odata folosit pixelul devine inutilizabil
        free_pixels[row:row + H, col:col + W] = -1

    # stergem padding ul
    img_mosaic = img_mosaic[:h, :w]

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape

    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    bigger_image = np.zeros((h + 2 * H, w + 2 * W, c), np.uint8)
    bigger_image[H: -H, W: -W, :] = params.image_resized.copy()

    first_row_start = 14
    row_index = 1

    for i in range(first_row_start, bigger_image.shape[0] - H, H):
        col_index = 0
        for j in range(0, bigger_image.shape[1] - W, W + 1 / 3 * W):
            patch = bigger_image[i:i + H, j:j + W]
            mean_patch = np.mean(patch, axis=(0, 1))
            index_ = get_sorted_indexes(params.mean_color_pieces, mean_patch)
            index = index_[0]

    return None
