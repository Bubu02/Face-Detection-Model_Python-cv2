import cv2
import numpy as np
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the directory containing known faces
dir_path = r"C:\Users\mebub_9a7jdi8\Desktop\Face Detection model\img"

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Lists to hold the images and labels
images = []
labels = []

# For each file in the directory
for i, file in enumerate(os.listdir(dir_path)):
    # Load the image in grayscale
    image = cv2.imread(os.path.join(dir_path, file), 0)

    # Add the image and label to the lists
    images.append(image)
    labels.append(i)

# Train the recognizer on the known faces
recognizer.train(np.array(images), np.array(labels))

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # For each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest
        roi_gray = gray[y:y+h, x:x+w]

        # Predict the ID of the face
        id_, _ = recognizer.predict(roi_gray)

        # Draw rectangle around the face and put text of the person's name (file name without extension)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, os.listdir(dir_path)[id_].split('.')[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # Display the output
    cv2.imshow('img', img)
    
    # Break if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()

