## -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:34:10 2021

@author: Jos de Wit (TU Delft, Department of Imaging Physics)
"""
import pymsgbox
import numpy as np
import matplotlib.pyplot as plt
from pypylon import pylon
try:
    import FWxC_COMMAND_LIB as fw
    import time
except OSError as ex:
    print("Warning:",ex)
from datetime import datetime
import os
from PIL import Image
import tkinter 
import  tkinter.messagebox
import socket

filter_slots=[[1,'BP540'],[2,'BP470'],[3,'BP505'],[4,'BP635'],[5,'BP695'],[6,'BP660']];
#%% initialize ethernet connection light source
def calc_checksum(s):
    sums = 0
    for c in s:
        sums += c
    sums = sums % 256
    return '%2X' % (sums & 0xFF)
    

def intensity_message(sourceno,intensity):
    if sourceno>2:
        print('invalid source number, choose from 0, 1, 2')
    if intensity>255:
        print('invalid intensity, choose an integer between 0 and 255, intensity set to 255')
        intensity=255
    if intensity<0:
        print('invalid intensity, choose an integer between 0 and 255, intensity set to 0')
        intensity=0
        
        
    message='@%02dF%03d'%(sourceno,intensity)
    message+=calc_checksum(message.encode('ascii'))
    message+='\r\n'
    return bytes(message,'utf-8')
TCP_IP = '192.168.0.16'#'145.94.204.69'#'192.168.0.16'  # Standard loopback interface address (localhost)
TCP_PORT = 30001        # Port to listen on (non-privileged ports are > 1023)

dst_IP = '192.168.0.2'  # Standard loopback interface address (localhost)
dst_PORT = 40001        # Port to listen on (non-privileged ports are > 1023)
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((dst_IP,dst_PORT))
UVonVISoff=[[0,0],[1,255],[2,255]]
UVoffVISon=[[0,255],[1,0],[2,0]]
#%% initialize filterwheel
devs = fw.FWxCListDevices()
if(len(devs)<=0):
   print('There is no filterwheel connected')
   exit()

FWxC = devs[0]
serialNumber=FWxC[0]

hdl = fw.FWxCOpen(serialNumber,115200,3)
#or check by "FWxCIsOpen(devs[0])"
if(hdl < 0):
    print("Connect ",serialNumber, "fail" )
    exit()
else:
    print("Connect ",serialNumber, "successful")

result = fw.FWxCIsOpen(serialNumber)
if(result<0):
    print("Open failed ")
    exit()
else:
    print("FWxC Is Open ")
speedmode=[0]
speedmodeList={0:"slow speed", 1:"high speed"}
result=fw.FWxCGetSpeedMode(hdl,speedmode)
if(result<0):
   print("Get Speed Mode fail",result)
else:
   print("Get Speed Mode :",speedmodeList.get(speedmode[0]))
#start position
result = fw.FWxCSetPosition(hdl, 2) 
#
#filterpositions=[[2,3,4,5],[5,4,3,2]]
filterpositions=[[1,2,3,4,5,6],[6,5,4,3,2,1]]
channel_index=[[1,2,3,4,5,0],[0,5,4,3,2,1]];
#exposure_times=[1e5,1.2e6] # [RBB, UV, UV_NIR2]
############################################################################################
exposure_times=[0.25e6,1.75e6] # [VIS, UV]
############################################################################################
#%% initialize camera
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()

cameraname=devices[0].GetFriendlyName()
lensname='C125-0818-5M-f:1.8/8mm'   

camera = pylon.InstantCamera()
camera.Attach(tl_factory.CreateFirstDevice())


#%% start imaging
date=datetime.today()
fdir=r'..\data\%04d%02d%02d'%(date.year,date.month,date.day)
if not os.path.isdir(fdir):
    os.mkdir(fdir)
imaging=True
for light_settings in UVoffVISon:
    s.send(intensity_message(light_settings[0],light_settings[1]))
while imaging:
    #%% define filename; set genotype via dialog box
    name=pymsgbox.prompt('What is the genotype name?')
    for i in range(1000):
        fname_base='BL33_'+name+'_%04d'%(i)
        if not os.path.isfile(os.path.join(fdir,fname_base+'_metadata.txt')):
            print(r'imaging file %s'%fname_base)
            break
    
    
    camera.Open()
    camera.ExposureTime.SetValue(1e4);
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
    if grab.GrabSucceeded():
        img = grab.GetArray()
    camera.StopGrabbing()
    VIS_imgs=np.empty(np.append(img.shape,6),dtype=np.uint8);
    
    UV_imgs=np.empty(np.append(img.shape,6),dtype=np.uint8);
#    NIR_imgs=np.empty(np.append(img.shape,3),dtype=np.uint8);
    #%% VIS image acquisition
    camera.ExposureTime.SetValue(exposure_times[0]);
    pymsgbox.alert('cabinet closed?')
    t0=time.time()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    for ii in range(len(filterpositions[0])):
        FW_pos = fw.FWxCSetPosition(hdl, filterpositions[0][ii])
        grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab.GrabSucceeded():
            img = grab.GetArray()
        VIS_imgs[:,:,channel_index[0][ii]]=img;
    camera.StopGrabbing()
    
    #%% UV image acquisition
    camera.ExposureTime.SetValue(exposure_times[1]);
    for light_settings in UVonVISoff:
        s.send(intensity_message(light_settings[0],light_settings[1]))
#
#    # last NIR UV image
#    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
#    grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
#    if grab.GrabSucceeded():
#        NIR_UV_img = grab.GetArray()
#    camera.StopGrabbing()
#    
#    camera.ExposureTime.SetValue(exposure_times[1]);
#    time.sleep(0.2)
    #camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    for ii in range(len(filterpositions[1])):
        FW_pos = fw.FWxCSetPosition(hdl, filterpositions[1][ii])
    #    time.sleep(3)
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        grab = camera.RetrieveResult(20000, pylon.TimeoutHandling_Return)
        if grab.GrabSucceeded():
            img = grab.GetArray()
        camera.StopGrabbing()
        UV_imgs[:,:,channel_index[1][ii]]=img;    
    
    #%% save data
    for light_settings in UVoffVISon:
        s.send(intensity_message(light_settings[0],light_settings[1]))
    
    plt.figure(1,figsize=[14,6])
    plt.subplot(1,2,1)
    plt.imshow(VIS_imgs[:,:,0:3])
    plt.title(fname_base+' white')
    plt.subplot(1,2,2)
    plt.imshow(UV_imgs[:,:,0:3])
    plt.title(fname_base+' UV')
    plt.show()
    
    print('saving data')
    img = Image.fromarray(VIS_imgs[:,:,0:3], 'RGB')
    img.save(os.path.join(fdir,fname_base+'_VIS_channel_0-2.tif'))
    img = Image.fromarray(VIS_imgs[:,:,3:6], 'RGB')
    img.save(os.path.join(fdir,fname_base+'_VIS_channel_3-5.tif'))
    
    img = Image.fromarray(UV_imgs[:,:,0:3], 'RGB')
    img.save(os.path.join(fdir,fname_base+'_UV_channel_0-2.tif'))
    img = Image.fromarray(UV_imgs[:,:,3:6], 'RGB')
    img.save(os.path.join(fdir,fname_base+'_UV_channel_3-5.tif'))
    
   
    file1 = open(os.path.join(fdir,fname_base+'_metadata.txt'),"w")
    
    file1.write("metadata of images of "+fname_base+"\n")
    dateTimeObj = datetime.now()
    st = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")
    file1.write("time: "+st+"\n")
    file1.write('VIS exposure time = %d us\n'%(exposure_times[0]))
    file1.write('UV exposure time = %d us\n'%(exposure_times[1]))
    for i in range(len(filter_slots)):
        file1.write('filter %d = %s\n'%(filter_slots[i][0],filter_slots[i][1]))
    for i in range(len(channel_index[0])):
        file1.write('VIS channel %d = filter slot %d = %s\n'%(channel_index[0][i],filterpositions[0][i],filter_slots[filterpositions[0][i]-1][1]))
    for i in range(len(channel_index[1])):
        file1.write('UV channel %d = filter slot %d = %s\n'%(channel_index[1][i],filterpositions[1][i],filter_slots[filterpositions[1][i]-1][1]))
    file1.write('UV light intensity = %d/255 and %d/255\n'%(UVonVISoff[1][1],UVonVISoff[2][1]))
    file1.write('VIS light intensity = %d/255 \n'%(UVoffVISon[0][1]))
    file1.write('camera = '+cameraname+'\n')
    file1.write('lens = '+lensname)
    file1.close() #to change file access modes
    print('it took %.1f s to image one instance'%(time.time()-t0))
    root=tkinter.Tk()
    imaging=tkinter.messagebox.askyesnocancel(title='proceed?', message='Images saved. Do you want to image another plant?')
    root.destroy()
    
#%% closing camera and filterwheel
camera.Close()
fw.FWxCClose(hdl)
for sourceno in range(3):
    s.send(intensity_message(sourceno,0))
s.close()