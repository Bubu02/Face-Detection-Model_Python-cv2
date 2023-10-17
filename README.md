# Face Recognition Project

## Overview
This project uses OpenCV's LBPH (Local Binary Patterns Histograms) Face Recognizer to perform face recognition on live video feed from your webcam and stores all the data in a output file.

## Dependencies
- Python 3
- OpenCV (`opencv-python` and `opencv-contrib-python` packages)

## How to Run
1. Organize your images in a directory such that each person's images are in a separate subdirectory. The name of the subdirectory should be the person's name. The original subdirectories containing images have been deleted for privacy reasons, so you can add your own images with respective names as the subdirectory names are used as labels.
2. Replace `C:\Users\mebub_9a7jdi8\Desktop\Face Detection model\img"` in the script with the path to your directory containing the subdirectories of images.
3. Run the script. It will load each image in each subdirectory into a list, and assign a label (the name of the image file without its extension) to each image.
4. During live face recognition from your webcam feed, it will display this label on the rectangle around each detected face.
5. When a face is detected for the first time, it will append this label and the current time to a text file named `output.txt`.

## Note
This script assumes that each image file contains exactly one face and that all faces are approximately the same size and are upright. If this is not the case, you may need to preprocess your images before using them for face recognition.

This is a basic model for face detection and might not work well in all scenarios or with all types of images. For better results, stay tuned for deep learning models on my profile.

The subdictionaries containing images were deleted for privacy. So users have to add their own images with respective name as the subdictionary names were used as lebels.

Please remember that working with images and especially with face detection and recognition has ethical considerations. Always ensure that you have permission to use and analyze the images and respect privacy.

