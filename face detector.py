import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def FaceBoundingBox():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow('Face Detector', frame)

        key = cv2.waitKey(1)

        if key == ord('r'):
            cv2.imwrite('../image.png', frame)
            video.release()
            cv2.destroyAllWindows()
            return x, y, w, h, cv2.imread('../image.png')


def LBP(gray):
    LBPimage = gray.copy()

    h, w = gray.shape
    # adding une extension
    image = np.insert(gray, w, gray[:, w - 1], axis=1)
    image = np.insert(image, 0, image[:, 0], axis=1)
    image = np.insert(image, h, image[h - 1, :], axis=0)
    image = np.insert(image, 0, image[0, :], axis=0)

    # getting le voisinage
    # rows
    for i in range(1, h + 1):
        # cols
        for j in range(1, w + 1):
            voisinage = image[i - 1:i + 2, j - 1:j + 2]
            # print("\nPour le pixel: ", image[i][j], ": \n", voisinage, )
            # SEUILLAGE et Affectation des Poids
            seuillage = [[1, 2, 4], [128, 0, 8], [64, 32, 16]]
            somme = 0
            for x in range(3):
                for y in range(3):
                    if voisinage[x][y] >= image[i][j]:
                        somme = somme + seuillage[x][y]
            # affecter la nouvelle valeur au pixel d'origine
            LBPimage[i - 1][j - 1] = somme
    count = [0] * 256
    m, n = LBPimage.shape
    for i in range(0, m - 1):
        for j in range(0, n - 1):
            count[LBPimage[i, j] - 1] += 1
    plt.figure()
    plt.title("LBPimage")
    plt.xlabel("intensité du gris")
    plt.ylabel("nb pixels")
    plt.bar([i for i in range(0, 256)], count)
    plt.savefig("../LBPimage" + '.png')
    return cv2.imread("../LBPimage" + '.png')


def HOG(Image):
    angles = []

    Image = np.insert(Image, 12, Image[:, 12 - 1], axis=1)
    Image = np.insert(Image, 0, Image[:, 0], axis=1)
    Image = np.insert(Image, 12, Image[12 - 1, :], axis=0)
    Image = np.insert(Image, 0, Image[0, :], axis=0)

    for i in range(1, 13):
        # cols
        for j in range(1, 13):
            grad = int(Image[i + 1][j]) - int(Image[i - 1][j]), int(Image[i][j + 1]) - int(Image[i][j - 1])
            if grad[1] != 0:
                angle = math.floor(math.degrees(math.atan(grad[0] / grad[1])))
                if angle < 0: angle = angle + 360
                angles.append(angle)

    count = [0] * 8

    for i in range(0, len(angles)):
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

    plt.figure()
    plt.title("Histogramme")
    plt.xlabel("les angles")
    plt.ylabel("Pixels")
    # ["-22.5 - 22.5", "22-67", "67-112", "112-157", "157-202", "202-247", "247-292", "292-338"]
    plt.bar([0, 1, 2, 3, 4, 5, 6, 7], count)
    plt.savefig('../HOG.png')
    return cv2.imread('../HOG.png')


# x, y, w, h, image = FaceBoundingBox()
# # cv2.imshow('Face', image)
# face = image[y+1:y+h , x+1:x+w]
# face = cv2.resize(face, (128,128), interpolation = cv2.INTER_AREA)
# cv2.imwrite('../dataset/face08.png', face)

face = cv2.imread('../face.png', 0)
cv2.imshow('Face', face)

# LBP
LBP_Face = LBP(face)
# HOG
hog_Hist = HOG(face)

cv2.imshow('LBP', LBP_Face)
cv2.imshow('HOG', hog_Hist)

cv2.waitKey(0)
