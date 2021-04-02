from Algorithms import *

seuil_lbp = 700
seuil_hog = 16
seuil_moy = 50


# Opens camera, returns a (not normalized) grayscale image when u press r
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


# opens camera, crops the image(BB), normalizes the image, retruns just the face
def getFace(normalized):
    # To get face from a camera
    x, y, w, h, image = FaceBoundingBox()
    face = image[y + 1:y + h, x + 1:x + w]
    # resize image to normalized scale value
    face = cv2.resize(face, (normalized, normalized), interpolation=cv2.INTER_AREA)
    cv2.imwrite('../Newface.png', face)
    return face


# reads image, detects faces, crops the image , returns face in grayscale
def loadImage(path, normalized):
    face = cv2.imread(path, 0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    for x, y, w, h in faces:
        face = cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 1)
        break
    face = face[y + 1:y + h, x + 1:x + w]
    face = cv2.resize(face, (normalized, normalized), interpolation=cv2.INTER_AREA)
    cv2.imshow('Face', face)
    return face


# opens camera, detects faces, apply HOG,LBP , save images/all values to dataset
def newImageForDataset(normalized, f_lbp, f_hog):
    x, y, w, h, image = FaceBoundingBox()
    face = image[y + 1:y + h, x + 1:x + w]
    face = cv2.resize(face, (normalized, normalized), interpolation=cv2.INTER_AREA)
    Histo_LBP = LBP(face, normalized, f_lbp)[2]
    Histo_HOG = HOG(face, normalized, f_hog)[1]

    # Saving LBP values
    with open('../dataset/dataLBP.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        file.write(str(Histo_LBP[0]))
        for item in Histo_LBP[1:]:
            file.write("," + str(item))
        file.write("\n")
        print("Saving LBP values")
    with open('../dataset/dataHOG.csv', 'a+', newline='') as file:
        file.write(str(Histo_HOG[0]))
        for item in Histo_HOG[1:]:
            file.write("," + str(item))
        file.write("\n")
        print("Saving HOG values")

    # Saves The image
    num_files = sum(os.path.isfile(os.path.join("../dataset", f)) for f in os.listdir("../dataset"))
    cv2.imwrite("../dataset/face" + str(num_files - 2) + ".png", face)
    print("Image is now saved to dataset")


# Compares LBP/HOG values of given face to faces in dataset
def recognize(face, normalized, f_lbp, f_hog):
    Imageinput_LBPvalues = LBP(face, normalized, f_lbp)[2]
    Imageinput_HOGvalues = HOG(face, normalized, f_hog)[1]

    # compare values with dataset
    with open('../dataset/dataLBP.csv', 'rt') as f1, open("../dataset/dataHOG.csv") as f2:
        print("comparing LBP and HOG values with values from dataset")

        # get liste des valeurs du dataset
        lbp_dataset = list(csv.reader(f1))
        hog_dataset = list(csv.reader(f2))
        nb_imgs = len(lbp_dataset)
        match = 0
        moy = 0
        moy_lbp = 0
        moy_hog = 0

        # pour chaque image du dataset
        for image in range(nb_imgs):
            ref_lbp = [int(i) for i in lbp_dataset[image]]
            ref_hog = [int(i) for i in hog_dataset[image]]

            # calcul la distance entre l'image input et l'image du dataset
            distance_lbp = np.linalg.norm(np.array(ref_lbp) - np.array(Imageinput_LBPvalues))
            distance_hog = np.linalg.norm(np.array(ref_hog) - np.array(Imageinput_HOGvalues))

            moy += (distance_lbp / distance_hog) / nb_imgs
            moy_lbp += distance_lbp / nb_imgs
            moy_hog += distance_hog / nb_imgs

            if (moy <= seuil_moy) or (moy_lbp <= seuil_lbp and moy_hog <= seuil_hog):
                match += 1

        print("Moyennes: ", moy, moy_lbp, moy_hog, "\nMatches: ", match, "/", nb_imgs)

        if (moy <= seuil_moy) or (match >= nb_imgs / 2) or (moy_lbp <= seuil_lbp and moy_hog <= seuil_hog and match >= nb_imgs / 2):
            print("MATCH")
            return True
        else:
            print("NOT A MATCH")
            return False


def RealTime(normalized, f_lbp, f_hog):
    found = False
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)

    while not found:
        check, frame = video.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2, minSize=(100, 100))
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow('Face Detector', frame)
            cv2.waitKey(1)
            frame = frame[y + 1:y + h, x + 1:x + w]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (normalized, normalized), interpolation=cv2.INTER_AREA)
            if recognize(frame, normalized, f_lbp, f_hog):
                found = True
                video.release()
                cv2.destroyAllWindows()


def RealTime(normalized, f_lbp, f_hog):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)


    while True:
        check, frame = video.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2, minSize=(100, 100))
        if len(faces) > 0:
            for x, y, w, h in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow('Face Detector', frame)

            face = frame[y + 1:y + h, x + 1:x + w]
            # resize image to normalized scale value
            face = cv2.resize(face, (normalized, normalized), interpolation=cv2.INTER_AREA)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            match = recognize(face,normalized,f_lbp,f_hog)
            if match:
                video.release()
                cv2.destroyAllWindows()
                break


def main():
    # PARAMETRES
    normalized = 128
    f_lbp, f_hog = 8, 6

    # ----------------------REAL TIME DETECTION------------------------
    RealTime(normalized, f_lbp, f_hog)

    # --------------RECOGNIZE FACE FROM CAMERA/IMAGE------------------
    # face = getFace(normalized)  # Camera
    # face = loadImage("face.png", normalized)  # Existing Image
    # recognize(face, normalized, f_lbp, f_hog)

    # ---------------IMPORT NEW IMAGE TO DATASET -------------------------------
    # newImageForDataset(normalized, f_lbp, f_hog)


if __name__ == "__main__":
    main()
    print("Done")

# cv2.waitKey(0)
