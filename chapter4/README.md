# UV-fluorescence imaging of lettuce enables pre-symptomatic detection and quantification of downy mildew disease

### This repository contains the scripts used to process the images and generate the data presented in chapter 4 of my PhD thesis.

UV-fluorescence and white light images were captured in the TUD imaging cabinet, which was controled with the script _TUD_imager.py_.
The images were processed and analysed in two different ways:

1. Image histogram statistics to quantify color variation with
- histogram-stats_threeleaves.py for the images from the QTL mapping experiment
- histogram-stats_singleleaf.py for the time-series images and the experiment comparing image-based quantification with qPCR.

2. U-NET semantic segmentation to quantify blue-green fluorescent area with
- combine_labels.ipynb to generate training labels
- resize.ipynb to resize all used images
- unet.ipynb to train and apply the CNN based on U-NET



