import cv2 
import os
from datetime import datetime
import numpy as np
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the directory containing the subdirectories of images
dir_path = r"C:\Users\mebub_9a7jdi8\Desktop\Face Detection model\img"

# Initialize empty lists to hold the images and labels
images = []
labels = []

# Setting a threshold value
thresholdValue = 127  # This is just an example value
maxVal = 255  # This is just an example value

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

# Initialize a dictionary to hold start times for detected faces
start_times = {}

# Initialize a dictionary to hold end times for detected faces
end_times = {}

# Initialize a set to hold labels that have been written to file
written_labels = set()

# Open the output text file
with open('output.txt', 'a') as f:
    while True:
        # Read the frame
        _, img = cap.read()

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Set a threshold
        _, thresholded = cv2.threshold(gray, thresholdValue, maxVal, cv2.THRESH_BINARY)

        # Detect faces on the thresholded image
        faces = face_cascade.detectMultiScale(thresholded, 1.1, 4)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # For each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest
            roi_gray = gray[y:y+h, x:x+w]

            # Predict the ID of the face
            id_, conf = recognizer.predict(roi_gray)

            if labels[id_] not in start_times:
                start_times[labels[id_]] = time.time()
            end_times[labels[id_]] = time.time()  # Update end time whenever this face is detected

            if end_times[labels[id_]] - start_times[labels[id_]] > 15 and labels[id_] not in written_labels:  # If this face has been detected continuously for more than 3 seconds and has not been written to file yet
                f.write(f'{labels[id_]} was present at {datetime.now()}\n')  # Write to file
                written_labels.add(labels[id_])  # Add label to written_labels set
                print(f'{labels[id_]} is detected.')

            # Draw rectangle around the face and put text of the person's name (file name without extension)
            if conf < 40:  # You can adjust this confidence threshold as needed
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, labels[id_], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # Display the output
        cv2.imshow('img', img)
        
        # Break if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

# Release the VideoCapture object
cap.release()
