# File name: image-prep.py
# Project: trypan-blue-lettuce-CNN

# Author: Andrew Pape
# Guido van den Ackervecken Lab
# Plant-Microbe Interactions Group
# Universiteit Utrecht

# Attribution:
#   for isolating the largest object, code was adapted from user "Andriy Makukha"
#        at https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv

#   for tiling the images, code was adapted from user "Ivan"
#        at https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python

#   for determining if a tile was within a circular area of interest, code was adapted from user "sabiland"
#        at https://stackoverflow.com/a/48083218

#Python v3.8.5
import os
import itertools as itertools
import numpy as np # v1.19.2
import cv2 as cv # v4.5.3
from PIL import Image # v7.2.0

Image.MAX_IMAGE_PIXELS = None # Disable built-in DOS attack protection of PIL to accomodate large input image size

# Create function 'tile' to perform on imagees of whole leaf discs ahead of model training
# Broadly, the following steps occur in this script:
# 1. Find the leaf disc in the image and enclose it in a bounding box.
# 2. Define a circle around the centerpoint of that bounding box. This is the region of interest. 
# 3. Divide the pixels inside that circle into separate tiles. 
# 4. Save all tiles to an output directory from which 'S2_image-sorter2.py' will draw during classification steps. 

# Define the function 'tile' with the following parameters:

def tile(filename,          #'filename' = the image to be tiled.
        input_directory,    #'input_directory' = the location of all the images to be tiled.
        output_directory,   #'output_directory' = the destination for tiles that are to be used in training the CNN.
        overflow_dir,       #'overflow_directory' = the destination for tiles are outside the area of interest.
        check_dir,          #'check_dir' = the destination for original images with the rectangular and circular areas of interest annotated.
        thresh_value = 200, #'thresh_value' = the threshold value to use in order to discriminate contours.
        tile_size = 400,    #'tile_size' = the length (in pixels) of one side of a tile.
        radius = 5800):     #'radius' = the radius (in pixels) of the circular area of interest within the leaf disc.

 
    name, ext = os.path.splitext(filename)
    #extract the file name and the file extension from the file

    img_PIL = Image.open(os.path.join(input_directory, filename))
    img_cv = cv.imread(os.path.join(input_directory, filename))
    #define object 'img_PIL' as the image to be tiled, open the image file.
    #define object 'img_cv' as the image to be interpreted as a numpy array and analyzed for contours

    gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)
    # convert the RGB image 'img' to grayscale.

    retval, thresh_gray = cv.threshold(gray, thresh = thresh_value, maxval=255, type=cv.THRESH_BINARY_INV)
    # apply threshold to the inverted grayscale 'img'.

    contours, hierarchy = cv.findContours(thresh_gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # use the threshold to generate a list of all the objects in 'img'.

    # Find the object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours: #iterate through the list of objects to find the biggest one.
        x,y,w,h = cv.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx 
    # mx = the coordinates that correspond to the rectangular bounding box that encompasses the largest object
        #(i.e. the leaf disc)

    centerCoord = (mx[0]+(mx[2]/2), mx[1]+(mx[3]/2))
    # Calculate x,y coordinate of the center of the largest contour in 'img_cv'.
    
    cX = int(centerCoord[0])
    cY = int(centerCoord[1])
    # Create objects 'cX' and 'cY'to contain the X and Y coordinate of the ROI center, respectively.

    grid = list(itertools.product(range(y, h - h%tile_size, tile_size), range(x, w-w%tile_size, tile_size)))
        # Define the grid lines (as a list) within the range of the largest contour
        # Along which the tiling cuts will be made.
        # Note: partial tiles are ignored

    for i, j in grid: #iterate through the list 'grid' 

        tile = (j, i, j + tile_size, i + tile_size)
        # use the 'grid' list to make a tile with dimensions according to 'tile_size' argument
        
        dx = max(abs(cX - j), abs((j+tile_size)- cX))
        dy = max(abs(cY - i), abs((i+tile_size)- cY))
        # Determine how far away the farthest point of each tile is from the center of the area of interest

        if radius*radius >= (dx * dx) + (dy * dy):
            # Use the Pythagorean theorum to determine if the furthest point of each tile
            #   is within the circular area of interest. 
              
            out = os.path.join(output_directory, f'{name}_{i}_{j}{ext}')        
            # Output path to 'output_directory' argument, output file naming regime according to 'grid',
            #   plus preserved extension.
            # These are the tiles within the circular area of interest

            img_PIL.crop(tile).save(out)
            # Make the cuts on the input image according to the tile iteration
            #     and save the tile in the 'output_directory' according to the naming regime.

        else: #else send the tile to the overflow_dir
            out_over = os.path.join(overflow_dir, f'{name}_{i}_{j}{ext}')
             # Output path to 'overflow_dir' argument, output file naming regime according to 'grid', plus preserved extension

            img_PIL.crop(tile).save(out_over)
            # Make the cuts on the image according to the tile iteration
            #   and save the tile in the 'overflow_dir' according to the naming regime. 
            # These are the tiles that are outside of our circular area of interest
            #   but are preserved as a check.
        
    cv.circle(img_cv, (cX, cY), radius, (200, 0, 0), 2)
    cv.rectangle(img_cv,(x,y),(x+w,y+h),(200,0,0),2)
    cv.imwrite(os.path.join(check_dir, f'{name}_{ext}'), img_cv)
    # Annotate the original image with the region of interest and the bounding box.
    # Save the annotated image to the 'check_dir' parameter.
    # These annotated images can be used to visualize the area of interest that was determined.
    # If results are sub-optimal, make adjustments to the 'thresh_value' parameter.


# Call the 'tile' function and loop through the input directory 
# For all the images in 'input_directory' argument, perform the 'tile' function
samples = []
files = os.listdir('model-test-disc-set_2')
for i in files:
    samples.append(i)

for i in samples:
    tile(filename = i,
        input_directory = 'model-test-disc-set_2',
        output_directory = 'model-test-set_2-ROI-tiles',
        overflow_dir = 'S1_overflow-tiles',
        check_dir = 'model-test-set_2-check-ROI',
        thresh_value = 200,
        tile_size = 400,
        radius = 5800)