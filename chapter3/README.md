# Microscopic image analysis of lettuce downy mildew infection using a convolutional neural network

## This repository contains all the scripts used to process the images and generate the data presented in chapter 3 of my PhD thesis.

Using a Zeiss Axiozoom microscope and the Zeiss Zen Blue software we generated high resolution images (e.g. ca. 15,000 pixels diameter) of trypan blue stained lettuce leaf discs (1 cm diameter) infected with downy mildew. The details are described in the thesis.
The scripts in this repository are meant to:

* slice leaf disc images into 400 x 400 pixel patches (01_prep.py)
* sort the image patches into classes (infected, not infected) (02_image-sorter2.py)
* train and test a convolutional neural network (03_train-test.ipynb)
* apply the trained convolutional neural network to classify 400 x 400 pixel patches in new leaf disc images (04_deploy.ipynb)

I ran steps 1 and 2 locally on my Macbook, uploaded the folders with sorted image patches as ZIP file to Google Drive and then ran steps 3 and 4 in Google Colab. Also the input images for step 4 are saved and directly processed on Google Drive.
  
![overview](https://github.com/sebastiantonn/phd/assets/90251517/ce74c725-33dd-4dfa-a384-018e67814a42)

