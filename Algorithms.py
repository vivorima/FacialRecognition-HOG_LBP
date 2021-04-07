import csv
import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Apply LBP on entire image (using LBP_window)
def LBP(face, normalized, f_lbp):
    windows_lbp = []
    histos_lbp = []
    # pour chaque window faire lbp
    for h in range(int(normalized / f_lbp)):
        sh = h * f_lbp
        fh = sh + f_lbp
        for w in range(int(normalized / f_lbp)):
            sw = w * f_lbp
            fw = sw + f_lbp
            window = face[sw:fw, sh:fh]
            window, histo = LBP_Window(window)
            windows_lbp.append(window)
            histos_lbp.append(histo)
            # cv2.imshow("Histogramme de la window", histogramme(histo))
            # cv2.rectangle(face_clone, (sw,sh), (fw,fh), (255, 255, 255), 1)
            # cv2.imshow("f",face_clone)
            # cv2.waitKey(1)
    # saveImages("LBP", windows_lbp)
    return windows_lbp, histos_lbp, concat(histos_lbp)


# Apply HOG on entire image (using HOG_window)
def HOG(face, normalized, f_hog):
    histos_hog = []
    for h in range(int(normalized / f_hog)):
        sh = h * f_hog
        fh = sh + f_hog
        for w in range(int(normalized / f_hog)):
            sw = w * f_hog
            fw = sw + f_hog
            window = face[sw:fw, sh:fh]
            hg = HOG_Window(window)
            histos_hog.append(hg)
            # cv2.imshow("histogramme de la window", histogramme(hg))
            # cv2.rectangle(face_clone, (sw, sh), (fw, fh), (255, 255, 255), 1)
            # cv2.imshow("f", face_clone)
            # cv2.waitKey(1)
            return histos_hog, concat(histos_hog)


def LBP_Window(gray_img):
    '''Applique LBP sur une image donnée '''
    lbp_image = gray_img.copy()
    image = RajouterExtension(gray_img)
    h, w = image.shape  # 10,10

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # getting le voisinage
            voisinage = image[i - 1:i + 2, j - 1:j + 2]
            # SEUILLAGE et Affectation des Poids
            seuillage = [[1, 2, 4], [128, 0, 8], [64, 32, 16]]

            somme = 0
            for x in range(3):
                for y in range(3):
                    if voisinage[x][y] >= image[i][j]:
                        somme = somme + seuillage[x][y]
            # affecter la nouvelle valeur au pixel d'origine
            lbp_image[i - 1][j - 1] = somme

        h, w = lbp_image.shape
        # HISTOGRAMME
        count = [0] * 256
        for i in range(h):
            for j in range(w):
                count[lbp_image[i, j] - 1] += 1

    return lbp_image, count


def HOG_Window(Image):
    angles = []
    hog = Image.copy()

    hog = RajouterExtension(hog)
    h, w = hog.shape  # 8.8

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            grad = int(hog[i + 1][j]) - int(hog[i - 1][j]), int(hog[i][j + 1]) - int(hog[i][j - 1])
            if grad[0] != 0:
                angle = math.floor(math.degrees(math.atan(grad[1] / grad[0])))
                if angle < 0:
                    angle = angle + 360
            else:
                angle = 0.1
            angles.append(angle)

    count = [0] * 8
    for i in range(len(angles)):
        if angles[i] <= 22 or angles[i] > 337:
            count[0] += 1
        elif 22 < angles[i] <= 67:
            count[1] += 1
        elif 67 <= angles[i] <= 112:
            count[2] += 1
        elif 112 < angles[i] <= 157:
            count[3] += 1
        elif 157 < angles[i] <= 202:
            count[4] += 1
        elif 202 < angles[i] <= 247:
            count[5] += 1
        elif 247 < angles[i] <= 292:
            count[6] += 1
        elif 292 < angles[i] <= 337:
            count[7] += 1

    return count


def RajouterExtension(gray):
    image = gray.copy()
    h, w = gray.shape
    # adding une extension
    image = np.insert(gray, w, gray[:, w - 1], axis=1)
    image = np.insert(image, 0, image[:, 0], axis=1)
    image = np.insert(image, h, image[h - 1, :], axis=0)
    image = np.insert(image, 0, image[0, :], axis=0)
    return image


def concat(lists):
    final = [0] * len(lists[0])
    for list in lists:
        for e in range(len(list)):
            final[e] += list[e]
    return final


#
def histogramme(count, titre, data):
    plt.figure()
    plt.title(titre)
    plt.xlabel(data)
    plt.ylabel("Number of Pixels")
    plt.bar([i + 1 for i in range(len(count))], count)
    plt.savefig('temp.png')
    return cv2.imread('temp.png')


# Saves images to dataset folder
def saveImages(folder, images):
    for w in range(len(images)):
        cv2.imwrite("../dataset/" + str(folder) + "/img" + str(w) + ".png", images[w])
