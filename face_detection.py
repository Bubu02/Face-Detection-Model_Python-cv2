import cv2
import os
from datetime import datetime
import numpy as np
# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the directory containing the subdirectories of images
dir_path = r"C:\Users\mebub_9a7jdi8\Desktop\Face Detection model\img"

# Initialize empty lists to hold the images and labels
images = []
labels = []

# For each subdirectory in the directory
for subdir in os.listdir(dir_path):
    # For each file in the subdirectory
    for file in os.listdir(os.path.join(dir_path, subdir)):
        # Load the image in grayscale
        image = cv2.imread(os.path.join(dir_path, subdir, file), 0)

        # Add the image and label (subdirectory name) to the lists
        images.append(image)
        labels.append(subdir)  # Here 'subdir' is used as label

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer on the known faces
recognizer.train(np.array(images), np.array(list(range(len(labels)))))

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize an empty set to hold the labels of detected faces
detected_faces = set()

# Open the output text file
with open('output.txt', 'a') as f:
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

            # If this face has not been detected before
            if labels[id_] not in detected_faces:
                # Add it to the set of detected faces
                detected_faces.add(labels[id_])

                # Write the label and current time to the text file
                f.write(f'{labels[id_]} was present at {datetime.now()}\n')

            # Draw rectangle around the face and put text of the person's name (file name without extension)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, labels[id_], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # Display the output
        cv2.imshow('img', img)
        
        # Break if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

# Release the VideoCapture object
cap.release()
