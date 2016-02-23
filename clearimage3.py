__author__ = 'Owner is the best'
import cv2 
import sys
import numpy as np
import os
from PIL import Image
#setting up the cascade path for windows and mac ( will be different depending on the computer)
cascadepath = "C:\\Users\Owner\Anaconda3\envs\\face\Library\etc\haarcascades/haarcascade_frontalface_default.xml"
macpath = "/anaconda/envs/face/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml" 
#recognizer and cascadesetup
faceCascade = cv2.CascadeClassifier(macpath)
recognizer = cv2.createFisherFaceRecognizer()
#function that grabs images(of faces) and returns 3 lisrs, one containing the labels, the names, and the last one is a list of arrays containing the pictures themselves
def getimagesandlabels(path):
    pathsandlabels = [os.path.join(path,f) for f in os.listdir(path)]
    print(pathsandlabels)
    files = []
    labels = []
    namess = []
    counter = 0
    for file in pathsandlabels:
        photo = Image.open(file).convert('L')
        image = np.array(photo, 'uint8')
        isthereface = faceCascade.detectMultiScale(image,scaleFactor=1.3,
        minNeighbors=3,
        minSize=(80, 80))
        for (x,y,h,w) in isthereface:
            cv2.imshow("Analyzing Faces...",image[y:y+h,x:x+w])
            files.append(image[y:y+h,x:x+w])
            if os.path.split(file)[1].split(".")[0] in namess:
                pass
            else:
                counter+=1
            namess.append(os.path.split(file)[1].split(".")[0])
            labels.append(counter)
            cv2.waitKey(50)
        print(counter)
    print(namess)
    return files,labels,namess
#setting up path for face identification
path = "faces"
files, labels,names = getimagesandlabels(path)
#training the recognizer with the faces folder
recognizer.train(files,np.array(labels))
print(np.array(labels))
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=3,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        rb, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print "Predicted",rb,"With COnfidence of...",confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if confidence >=30:
            cv2.putText(frame,str(names[rb-1]),(x,y),cv2.FONT_HERSHEY_COMPLEX
,3,(250,0,100))
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, everything closes
video_capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
