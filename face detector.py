import cv2;

def FaceBoundingBox():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in faces:
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
        
        cv2.imshow('Face Detector', frame)
        
        key = cv2.waitKey(1)

        if key == ord('r'):
            cv2.imwrite('image.png', frame)
            video.release()
            return x,y,w,h,frame

        if key == ord('x'):
            video.release()
            cv2.destroyAllWindows()
            break
    
    


x,y,w,h,image = FaceBoundingBox()
face = image[x:y, x+h:y+h]
cv2.imwrite('face.png', face)