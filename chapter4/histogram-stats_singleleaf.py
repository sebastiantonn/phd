#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:26:28 2022

@author: Sebastian Tonn, based on script by Mon-Ray Shao
Institute: Utrecht University
Department: Plant Microbe Interactions
Lab: Guido van den Ackerveken
"""

### This script takes images made with the TUD imager, corrects brightness with reference image,
### sets everything on the right of a defined x coordinate to black/0, segments the leaves, and generates histogram statistics.
### It is meant for images of single leaves taken using the "Otto-plant-holder",
### where we only want to get information from the one isolated leaf, but some parts of the rest of the plant are still
### in the image on the right hand side and have to be "cut" out.
### 

### Load packages
from PIL import Image
from tifffile import imread
import cv2 as cv2
import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal

def options():
    parser = argparse.ArgumentParser(description="UV image processing with OpenCV")
    parser.add_argument("-i", "--input", help="Input directory for images and reference", required=True)
    parser.add_argument("-o", "--output", help="Optional name of output directory saved within input directory (default = 'output'). ", required=False)
    parser.add_argument("-p", "--prefix", help="Optional prefix for output CSV files (default = 'channel'). ", required=False)
    parser.add_argument("-r", "--reference", help="Optional reference image for vignetting calibration, e.g. 'referencefilename_UV_0-2.tiff'", required=False)
    parser.add_argument("-n", "--NIR", help="Optional multiplication factor for NIR mask, e.g. -n 3", required=False)
    parser.add_argument("-e", "--erosion", help="Optional OpenCV mask erosion values and iterations, e.g. -e 1,1,1 (low erosion) or -e 6,6,1 (medium erosion)", required=False)
    parser.add_argument("-m", "--maxintensity", help="Optional max intensity cutoff value (default = 150)", required=False, type=int)
    parser.add_argument("-k", "--kernel", help="Kernel size for filter. Has to be an uneven number (default = 5)", required=False, type=int)
    parser.add_argument("-f", "--filter", help="Select filter type to apply, either 'gaussian' or 'median'. If not selected, no filter will be applied", required=False)
    args = parser.parse_args()
    return args

### Main workflow
def main():
    
    # Define user input parameters and create directories
    args = options()
    inputdir = args.input
    if args.output:
        output = args.output
        outputUV = os.path.join(output, 'UV')
        outputWL = os.path.join(output, 'WL')
        os.mkdir(outputUV)
        os.mkdir(outputWL)
    else: 
        output = os.path.join(inputdir, 'output')
    if not os.path.isdir(output):
        os.mkdir(output)
        
    outputUV = os.path.join(output, 'UV')
    outputWL = os.path.join(output, 'WL')
    os.mkdir(outputUV)
    os.mkdir(outputWL)
    
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = 'channel'
    
    # Create calibration values for vignetting and NIR intensity (if applicable)
    if args.reference:
        print('Correction using reference background selected')
        reference = os.path.join(inputdir, args.reference)
        ref_img = np.array(imread(reference))
        BP660ref = ref_img[:,:,0]
        BP660ref = BP660ref.astype('float')
        BP660ref = signal.spline_filter(BP660ref, lmbda = 10000)
        BP660ref[np.where(BP660ref < 0)] = 0
        relative = BP660ref / np.max(BP660ref)
    else:
        print('Correction using reference background not selected')
        relative = 1
        
    # Define intensity cutoff value to increase image brightness
    if args.maxintensity:
        maxint = args.maxintensity
        print('Intensity cut-off is set to', maxint)
    else:
        maxint = 150
        print('Intensity cut-off is set to 255 (default, for 8-bit images brightness will not be altered)')
        
    # Confirm filter type selection
    if args.filter:
        if args.filter in {'gaussian', 'median'}:
            print('Filter type to be applied:', args.filter)
            filtertype = args.filter
        else:
            print('Unvalid filter input! Filter must be either "gaussian" or "median". As default no filter will be applied' )
            filtertype = 'none'
    else:
        print('No filter will be applied.')
        filtertype = 'none'
        
    # Define kernel size for median filter. If no filter is selected, this argument will not be used.
    kernel = 5 # If no kernel is given in input, kernel size 5 will be used. If kernel is given as input, this will be overwritten in below if-statement
    if args.kernel and filtertype in {'gaussian', 'median'}: 
        if (args.kernel % 2) != 0: # Check if kernel is uneven number
            kernel = args.kernel
            print('Filter will be applied with kernel size', kernel)
        else:
            print('Invalid kernel size. Kernel must be uneven integer. Default kernel size 5 will be used')
            
    # Define multiplication factor for NIR channel image, to optimize segmentation   
    if args.NIR:
        n = float(args.NIR)
    else:
        n = 1

    # Extract raw data
    
    UVdata = []
    WLdata = []
    for filename in os.listdir(inputdir):
        if 'metadata' in filename and 'reference' not in filename:

            # Load images
            print('Processing sample', re.sub('_metadata.txt', '', filename))
            fname_UV1 = re.sub('metadata.txt', 'UV_channel_0-2.tif', filename)
            fname_UV2 = re.sub('metadata.txt', 'UV_channel_3-5.tif', filename)
            fname_WL1 = re.sub('metadata.txt', 'VIS_channel_0-2.tif', filename)
            fname_WL2 = re.sub('metadata.txt', 'VIS_channel_3-5.tif', filename)
            imageUV1 = np.array(imread(os.path.join(inputdir, fname_UV1)))
            imageUV2 = np.array(imread(os.path.join(inputdir, fname_UV2)))
            imageWL1 = np.array(imread(os.path.join(inputdir, fname_WL1)))
            imageWL2 = np.array(imread(os.path.join(inputdir, fname_WL2)))

            # Extract UV channels and multiply NIR channel
            UV_BP470 = imageUV1[:,:,2]
            UV_BP505 = imageUV2[:,:,0]
            UV_BP540 = imageUV1[:,:,1]
            UV_BP635 = imageUV2[:,:,1]
            UV_BP660 = imageUV1[:,:,0]
            UV_BP695 = imageUV2[:,:,2]
            UV_BP695 = np.rint(UV_BP695 * n)
            UV_BP695[np.where(UV_BP695 > 255)] = 255
            UV_BP695 = UV_BP695.astype('uint8')
            
            # Extract WL channels
            WL_BP470 = imageWL1[:,:,2]
            WL_BP505 = imageWL2[:,:,0]
            WL_BP540 = imageWL1[:,:,1]
            WL_BP635 = imageWL2[:,:,1]
            WL_BP660 = imageWL1[:,:,0]

            # Calibrate UV channels
            np.seterr(divide = 'ignore', invalid = 'ignore')
            UV_BP470 = UV_BP470 / relative
            UV_BP470[np.where(UV_BP470 > 255)] = 255
            UV_BP470 = UV_BP470.astype('uint8')
            UV_BP505 = UV_BP505 / relative
            UV_BP505[np.where(UV_BP505 > 255)] = 255
            UV_BP505 = UV_BP505.astype('uint8')
            UV_BP540 = UV_BP540 / relative
            UV_BP540[np.where(UV_BP540 > 255)] = 255
            UV_BP540 = UV_BP540.astype('uint8')
            UV_BP635 = UV_BP635 / relative
            UV_BP635[np.where(UV_BP635 > 255)] = 255
            UV_BP635 = UV_BP635.astype('uint8')
            UV_BP660 = UV_BP660 / relative
            UV_BP660[np.where(UV_BP660 > 255)] = 255
            UV_BP660 = UV_BP660.astype('uint8')
            
            # Increase brightness of blue and green channels for better visualization
            # (only for saving UV-RGB PNG images, the image statistics are still extracted from images with unaltered brightness)
            BP470bright = (UV_BP470 / maxint) * 255
            BP470bright = np.minimum(BP470bright, 255)
            BP470bright = BP470bright.astype('uint8')
            BP540bright = (UV_BP540 / maxint) * 255
            BP540bright = np.minimum(BP540bright, 255)
            BP540bright = BP540bright.astype('uint8')
            BP660image = UV_BP660.copy()
            
            # If given in input, apply filter to all three UV channels separately
            if filtertype == 'median':
                BP470bright = cv2.medianBlur(BP470bright, kernel)
                BP540bright = cv2.medianBlur(BP540bright, kernel)
                BP660image = cv2.medianBlur(BP660image, kernel)
            if filtertype == 'gaussian':
                BP470bright = cv2.GaussianBlur(BP470bright, (kernel,kernel), cv2.BORDER_DEFAULT)
                BP540bright = cv2.GaussianBlur(BP540bright, (kernel,kernel), cv2.BORDER_DEFAULT)
                BP660image = cv2.GaussianBlur(BP660image, (kernel,kernel), cv2.BORDER_DEFAULT)

            # Create UV-RGB image with brighter, and if selected filtered, blue and green channels for visualization
            UV_RGB = np.dstack([BP660image, BP540bright, BP470bright])
            # Create WL-RGB image with unaltered channels for visualization
            WL_RGB = np.dstack([WL_BP660, WL_BP540, WL_BP470])
            
            # Segment plant(s) and write to file for both UV and WL RGB images
            ret, mask = cv2.threshold(UV_BP695, 0, 255, cv2.THRESH_OTSU)
            if args.erosion:
                erode = args.erosion.split(',')
                mask = cv2.erode(mask, np.ones((int(erode[0]), int(erode[1]))), iterations = int(erode[2]))   
            
            UV_RGB[mask == 0] = 0
            WL_RGB[mask == 0] = 0
            
            # Save RGB files in the UV and WL directories respectively
            UV_corrected = Image.fromarray(UV_RGB)
            file_ending = '_0-2_'+filtertype+'_kernel'+str(kernel)+'_maxbright'+str(maxint)+'.png'
            if filtertype == 'none':
                file_ending = '_0-2_maxbright'+str(maxint)+'.png'
            UV_corrected.save(os.path.join(outputUV, re.sub('_0-2.tif', file_ending, fname_UV1)))
            
            WL_RGB = Image.fromarray(WL_RGB)
            WL_RGB.save(os.path.join(outputWL, re.sub('_0-2.tif', '_0-2.png', fname_WL1)))
            
            # Create array of all UV channels
            R = UV_BP660
            B = UV_BP470
            G = UV_BP540
            BGR = np.dstack([B,G,R])
            C = UV_BP505
            LR = UV_BP635
            HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
            H = HSV[:,:,0]
            S = HSV[:,:,1]
            V = HSV[:,:,2]
            LAB = cv2.cvtColor(BGR, cv2.COLOR_BGR2LAB)
            L = LAB[:,:,0]
            a = LAB[:,:,1]
            b = LAB[:,:,2]
            allChannels = np.dstack([R,G,B,C,LR,H,S,V,L,a,b])
            
            # Calculate histograms from every channel
            channelList = ['Red660','Green','Blue','Cyan','Red635','Hue','Sat','Val','L','a','b']
            
            row = {
                'Sample': re.sub('_metadata.txt', '', filename),
                'Leafpixels': np.count_nonzero(mask)}
            for i in range(0,11):
                
                if i == 5:
                    # calculate statistics for hue
                    img = allChannels[:,:,i]
                    
                    # shift hue by 90
                    n_shift = 90
                    img[:, :] = (img[:, :].astype(int) + n_shift) % 180
                    
                    row[channelList[i]+'_Mean'] = np.mean(img[mask==255])
                    row[channelList[i]+'_Median'] = np.median(img[mask==255])
                    row[channelList[i]+'_Mode'] = float(stats.mode(img[mask==255], keepdims = False)[0])
                    row[channelList[i]+'_Variance'] = np.var(img[mask==255], axis=None)
                    row[channelList[i]+'_Kurtosis'] = stats.kurtosis(img[mask==255], axis=None)
                    row[channelList[i]+'_Skewness'] = stats.skew(img[mask==255], axis=None)
                    
                else:
                    # calculate statistics fir all other channels
                    img = allChannels[:,:,i]
                    row[channelList[i]+'_Mean'] = np.mean(img[mask==255])
                    row[channelList[i]+'_Median'] = np.median(img[mask==255])
                    row[channelList[i]+'_Mode'] = float(stats.mode(img[mask==255], keepdims = False)[0])
                    row[channelList[i]+'_Variance'] = np.var(img[mask==255], axis=None)
                    row[channelList[i]+'_Kurtosis'] = stats.kurtosis(img[mask==255], axis=None)
                    row[channelList[i]+'_Skewness'] = stats.skew(img[mask==255], axis=None)
                    
            UVdata.append(row)
            
            # Create array of all WL channels (the UV data has already be appended, following code will overwright the variables with the WL data)
            R = WL_BP660
            B = WL_BP470
            G = WL_BP540
            BGR = np.dstack([B,G,R])
            C = WL_BP505
            LR = WL_BP635
            HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
            H = HSV[:,:,0]
            S = HSV[:,:,1]
            V = HSV[:,:,2]
            LAB = cv2.cvtColor(BGR, cv2.COLOR_BGR2LAB)
            L = LAB[:,:,0]
            a = LAB[:,:,1]
            b = LAB[:,:,2]
            allChannels = np.dstack([R,G,B,C,LR,H,S,V,L,a,b])
            
            # Calculate histograms from every channel
            channelList = ['Red660','Green','Blue','Cyan','Red635','Hue','Sat','Val','L','a','b']
            
            row = {
                'Sample': re.sub('_metadata.txt', '', filename),
                'Leafpixels': np.count_nonzero(mask)}
            for i in range(0,11):
                # calculate statistics
                img = allChannels[:,:,i]
                row[channelList[i]+'_Mean'] = np.mean(img[mask==255])
                row[channelList[i]+'_Median'] = np.median(img[mask==255])
                row[channelList[i]+'_Mode'] = float(stats.mode(img[mask==255], keepdims = False)[0])
                row[channelList[i]+'_Variance'] = np.var(img[mask==255], axis=None)
                row[channelList[i]+'_Kurtosis'] = stats.kurtosis(img[mask==255], axis=None)
                row[channelList[i]+'_Skewness'] = stats.skew(img[mask==255], axis=None)
            WLdata.append(row)
                
    dfUV = pd.DataFrame(UVdata)
    dfUV_outfile = os.path.join(outputUV, prefix + '_summary.csv')
    dfUV.to_csv(dfUV_outfile, index=False)
    
    dfWL = pd.DataFrame(WLdata)
    dfWL_outfile = os.path.join(outputWL, prefix + '_summary.csv')
    dfWL.to_csv(dfWL_outfile, index=False)


# Call program
if __name__ == '__main__':
    main()