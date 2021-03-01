import csv
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def newImageForDataset(normalized, f_lbp, f_hog):
    x, y, w, h, image = FaceBoundingBox()
    face = image[y + 1:y + h, x + 1:x + w]
    face = cv2.resize(face, (normalized, normalized), interpolation=cv2.INTER_AREA)
    Histo_LBP = LBP(face, normalized, f_lbp)[2]
    Histo_HOG = HOG(face, normalized, f_hog)[1]
    print("LBP and HOG... DONE")

    with open('../dataset/dataLBP.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        file.write(str(Histo_LBP[0]))
        for item in Histo_LBP[1:]:
            file.write("," + str(item))
        file.write("\n")
    with open('../dataset/dataHOG.csv', 'a+', newline='') as file:
        file.write(str(Histo_HOG[0]))
        for item in Histo_HOG[1:]:
            file.write("," + str(item))
        file.write("\n")
        print("HOG and LBP Values saved in dataset")

    num_files = sum(os.path.isfile(os.path.join("../dataset", f)) for f in os.listdir("../dataset"))
    cv2.imwrite("../dataset/face" + str(num_files - 2) + ".png", face)
    print("image added to dataset")


def FaceBoundingBox():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2, minSize=(100, 100))
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow('Face Detector', frame)

        key = cv2.waitKey(1)

        if key == ord('r'):
            cv2.imwrite('../image.png', frame)
            video.release()
            cv2.destroyAllWindows()
            return x, y, w, h, cv2.imread('../image.png', 0)


def concat(lists):
    final = [0] * len(lists[0])
    for list in lists:
        for e in range(len(list)):
            final[e] += list[e]
    return final


def saveImages(folder, images):
    for w in range(len(images)):
        cv2.imwrite("../dataset/" + str(folder) + "/img" + str(w) + ".png", images[w])


def RajouterExtension(gray):
    image = gray.copy()
    h, w = gray.shape
    # adding une extension
    image = np.insert(gray, w, gray[:, w - 1], axis=1)
    image = np.insert(image, 0, image[:, 0], axis=1)
    image = np.insert(image, h, image[h - 1, :], axis=0)
    image = np.insert(image, 0, image[0, :], axis=0)
    return image


def histogramme(count, titre, data):
    plt.figure()
    plt.title(titre)
    plt.xlabel(data)
    plt.ylabel("Number of Pixels")
    plt.bar([i + 1 for i in range(len(count))], count)
    plt.savefig('temp.png')
    return cv2.imread('temp.png')


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


def recognizeFromCamera(face, normalized, f_lbp, f_hog):
    Histo_LBP = LBP(face, normalized, f_lbp)[2]
    Histo_HOG = HOG(face, normalized, f_hog)[1]
    print("HOG and LBP done on input image")

    d = []
    # compare values with dataset
    with open('../dataset/dataLBP.csv', 'rt') as f1, open("../dataset/dataHOG.csv") as f2:
        print("comparing LBP and HOG values with dataset...")
        lbp = list(csv.reader(f1))
        hog = list(csv.reader(f2))
        lbp_values = []
        hog_values = []
        dist_LBP = 0
        dist_HOG = 0
        for image in range(len(lbp)):
            lbp_values = [int(i) for i in lbp[image]]
            hog_values = [int(i) for i in hog[image]]
            dist_LBP += np.linalg.norm(np.array(lbp_values) - np.array(Histo_LBP))
            dist_HOG += np.linalg.norm(np.array(hog_values) - np.array(Histo_HOG))

    dist_LBP = dist_LBP / len(lbp)
    dist_HOG = dist_HOG / len(lbp)
    print(dist_LBP, dist_HOG)
    if dist_LBP <= 700 and dist_HOG <= 20:
        print("it's Rima!")
    else:
        print("its not rima")


normalized = 128
f_lbp, f_hog = 8, 6
# To get face from a camera
x, y, w, h, image = FaceBoundingBox()
face = image[y + 1:y + h, x + 1:x + w]
face = cv2.resize(face, (normalized, normalized), interpolation=cv2.INTER_AREA)
cv2.imwrite('../Newface.png', face)
# ---------------------------------------------------
# face = cv2.imread('ines.jpg',0)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faces = face_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
# for x, y, w, h in faces:
#     face = cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 1)
#     break
# face = face[y+1:y+h , x+1:x+w]
# face = cv2.resize(face, (normalized,normalized), interpolation = cv2.INTER_AREA)
cv2.imshow('Face', face)

recognizeFromCamera(face, normalized, f_lbp, f_hog)
# newImageForDataset(normalized,f_lbp,f_hog)
cv2.waitKey(0)
