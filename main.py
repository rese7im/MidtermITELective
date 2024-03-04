import cv2
import numpy as np
import math
import time
import os

# initialize the video capture object
cap = cv2.VideoCapture(0)

#  Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set parameters for cropping and resizing the detected face
offset = 20
imgSize = 300

# set the path to save the images
path = 'Dataset/Face'
counter = 0

# create the directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

# set to detect faces as image
def detect_faces(image_path):
    # Read the image
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # set to detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around the faces and add labels
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # Display the result
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# loop statement
while True:
    # capture frame-by-frame
    ret, img = cap.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # process each
    for (x, y, w, h) in faces:
        # draw a bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # set a label
        cv2.putText(img, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # display the original frame
    cv2.imshow("Image", img)

    # functions as save
    key = cv2.waitKey(1)
    if key == ord("s"):  # Save the detected face image
        counter += 1
        filename = f'{path}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, img)
        print(f"Saved image: {filename}")
        print(counter)

    # exit
    if key == ord("i"):
        break

# close exit
cap.release()
cv2.destroyAllWindows()
