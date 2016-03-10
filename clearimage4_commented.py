__author__ = 'Owner'
#import libraries
import cv2 
import sys
import numpy as np
import os
from PIL import Image
#setting up the cascade path for windows and mac ( will be different depending on the computer)
cascadepath = "C:\\Users\Owner\Anaconda3\envs\\face\Library\etc\haarcascades/haarcascade_frontalface_default.xml"
macpath = "/anaconda/envs/face/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml" 
#create a cascade classifier object.
faceCascade = cv2.CascadeClassifier(macpath)
#create a FisherFaceRecognizer Object
recognizer = cv2.createFisherFaceRecognizer()
#path is where the folder with the faces used to train the recognizer is located. Since the .pu file is in the same place as the folder, stating just the name of the folder is more than enough
path = "faces"
#Define a function that trains the Fisher Face recognizer. It takes the path of where the faces are and trains the recognizer with said faces.
def getimagesandlabels(path):
    #gets the directory of each face
    pathsandlabels = [os.path.join(path,f) for f in os.listdir(path)]
    print(pathsandlabels)
    #files is a list of matrices that contain the RGB values of the pictures
    files = []
    #Labels is a list of the label of each person.Each person gets a unique label (e.g Michael could get #1 while Jon gets #2)
    labels = []
    #Namess is a list of the names of the people that were analyzed.
    namess = []
    #name dict is a dictionary that cotains the label of the person as the key and his/her name as the value.
    namedict = {}
    #Counter is used to set up the label
    counter = 0
    #for loop that acceses each image in the faces folder
    for file in pathsandlabels:
        #opens the image and converts it to greyscale
        photo = Image.open(file).convert('L')
        #Converts photo into an array in uint8, which is consituted of values from 0-255, the RGB values.
        image = np.array(photo, 'uint8')
        #resize pictures
        resized = photo.resize
        #run the detectMultiscale method, which will return an (x,y,h,w) value for each face analized in the given image
        isthereface = faceCascade.detectMultiScale(image,scaleFactor=1.3,
        minNeighbors=3,
        minSize=(80, 80))
        #analyzes each found face
        for (x,y,h,w) in isthereface:
            #converts array to image, resizes to 600x600, converts back to array
            current = image[y:y+h,x:x+w]
            convert = Image.fromarray(current)
            convert = convert.resize((600,600))
            current = np.array(convert,'uint8')
            cv2.imshow("Analyzing Faces...",current)
            #appends array to list
            files.append(current)
            #checks if the user (name) is already in the namess list. If it is not, it will increase the counter (label) by 1
            if os.path.split(file)[1].split(".")[0] in namess:
                pass
            else:
                counter+=1
            #appends label, name to their lists. Adds label as key and name as value to the namedict dictionary.
            namess.append(os.path.split(file)[1].split(".")[0])
            namedict[str(counter)] = os.path.split(file)[1].split(".")[0]
            labels.append(counter)
            cv2.waitKey(50)
        print(counter)
    print namess, namedict
    return files,labels,namess,namedict
#set variables as values of getimagesandlabels()
files, labels,namess,namedict = getimagesandlabels(path)
#training the recognizer with the faces folder
recognizer.train(files,np.array(labels))
print(np.array(labels))
#opens webcam
video_capture = cv2.VideoCapture(0)
#function to recognize faces and identify who they are
while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()
    #convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #run the detectmultiscale method
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=3,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #same as above
        current = gray[y:y+h,x:x+w]
        convert = Image.fromarray(current)
        convert = convert.resize((600,600))
        current = np.array(convert,'uint8') 
        #the predict method from the recognizer will give rb (who the person is, by label) and a confidence level
        rb, confidence = recognizer.predict(current)
        print "Predicted",rb,"With COnfidence of...",confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if confidence >=30:
            #this is where the dictionary is useful (link label to name)
            cv2.putText(frame,namedict[str(rb)],(x,y),cv2.FONT_HERSHEY_COMPLEX
,3,(250,0,100))
    # Display the resulting frame
    cv2.imshow('Video', frame)
    #program will exit if you press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, everything closes
video_capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
