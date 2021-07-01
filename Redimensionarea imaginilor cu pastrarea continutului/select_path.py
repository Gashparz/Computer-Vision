import sys
import numpy as np
import pdb
from parameters import *


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_greedy_path(E):
    col = np.argmin(E[0])
    linie = 0
    path = [(linie, col)]

    for i in range(E.shape[0]):
        linie = i
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            aux = np.array([E[linie][path[-1][1]], E[linie][path[-1][1] + 1]])
            opt = np.argmin(aux)
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            aux = np.array([E[linie][path[-1][1] - 1], E[linie][path[-1][1]]])
            opt_aux = np.argmin(aux)
            if opt_aux == 0:
                opt = -1
            else:
                opt = 0
        else:
            aux = np.array([E[linie][path[-1][1] - 1], E[linie][path[-1][1]], E[linie][path[-1][1] + 1]])
            opt_aux = np.argmin(aux)
            if opt_aux == 0:
                opt = -1
            elif opt_aux == 1:
                opt = 0
            else:
                opt = 1

        col = path[-1][1] + opt
        path.append((linie, col))

    return path


def select_dynamic_programming_path(params: Parameters, E):
    # construim o matrice de costuri cu dimensiunile matricei de energie
    cost_matrix = np.zeros((E.shape[0], E.shape[1]))
    # copiem primul rand
    cost_matrix[0] = E[0].copy()

    # calculam costul fiecarui pixel folosind formula data
    for i in range(1, E.shape[0]):
        for j in range(E.shape[1]):
            if j == 0:  # daca suntem pe prima coloana avem grija sa nu adunam in stanga
                cost_matrix[i][j] = E[i][j] + min(cost_matrix[i - 1][j], cost_matrix[i - 1][j + 1])
            elif j == E.shape[1] - 1:  # daca suntem pe ultima coloana avem grija sa nu adunam la dreapta
                cost_matrix[i][j] = E[i][j] + min(cost_matrix[i - 1][j - 1], cost_matrix[i - 1][j])
            else:
                cost_matrix[i][j] = E[i][j] + min(cost_matrix[i - 1][j - 1], cost_matrix[i - 1][j],
                                                  cost_matrix[i - 1][j + 1])

    # am adaugat un nou parametru functiei select_dynamic_programming_path pentru a putea accesa return ul lui selectROI
    if params.resize_option == 'eliminaObiect':
        x = params.x0
        y = params.y0
        w = params.w
        h = params.h

        # pentru a forta algorimul sa decupeze sectiunea selectata facem in stanga si in dreapta ei elemente foarte mari
        for i in range(cost_matrix.shape[0]):
            cost_matrix[i][:y-1] = 99999999
            cost_matrix[i][y+w+1:] = 99999999

    # aflam care este cel mai mic element de pe ultima linie (punctul de pornire pentru a construi drumul in sus)
    col = np.argmin(cost_matrix[E.shape[0] - 1])
    linie = E.shape[0] - 1
    path = [(linie, col)]

    # trecem prin matrice pornind de la ultima linie in sus
    for i in reversed(range(cost_matrix.shape[0] - 1)):
        linie = i
        col_old = path[-1][1]

        if col_old == 0:
            # pixelul este localizat la marginea din stanga avem grija sa nu luam in considerare si
            # pixelul din stanga acestuia
            aux = np.array([cost_matrix[linie][col_old], cost_matrix[linie][col_old + 1]])
            # in functie de rezultat ne mutam cu coloana spre opt
            opt = np.argmin(aux)
        elif col_old == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            # avem grija sa nu luam in considerare pixelul din dreapta
            aux = np.array([cost_matrix[linie][col_old - 1], cost_matrix[linie][col_old]])
            opt_aux = np.argmin(aux)
            if opt_aux == 0:
                opt = -1
            else:
                opt = 0
        else:
            aux = np.array(
                [cost_matrix[linie][col_old - 1], cost_matrix[linie][col_old], cost_matrix[linie][col_old + 1]])
            opt_aux = np.argmin(aux)
            if opt_aux == 0:
                opt = -1
            elif opt_aux == 1:
                opt = 0
            else:
                opt = 1
        path.append((linie, col_old + opt))

        # dam reverse la drum deoarece am inceput de jos
        path_b = path[::-1]
    return path_b


def select_path(params: Parameters, E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(params, E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)
