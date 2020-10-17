#!/usr/bin/env python
# coding: utf-8

# In[1]:


# All the nessesary libraries for this work 
import numpy as np
from scipy import ndimage as img
from scipy import io as sio
import matplotlib.pyplot as plt
import pyshearlab
import scipy.misc
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.signal import medfilt
from scipy import arange
import sys
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import scipy as sc
from scipy.ndimage import gaussian_filter
from numpy import vstack
from numpy import hstack
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pywt
from sklearn import preprocessing
from scipy.fftpack import fft, ifft
from scipy import fftpack
from scipy.stats import norm
import scipy as sc
import statistics
from scipy import signal
from scipy import stats
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2
from scipy import ndimage as img
from scipy import io as sio
import matplotlib.pyplot as plt
import pyshearlab
import scipy.misc
import cv2
import imageio
from collections import Counter
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.stats import chi2
plt.style.use('dark_background')


# Asking user to .brw file directory to be analysed, creating two files
# "Denoised" - file which is the result of CWT + PCA algorithm
# "DenoisedAndSheared" - file which is the final result of CWT + PCA + Shearlets
# In[2]:
print("Enter file directory to be denoised")
fileName = input()

print("Enter PCA threshold in number of standard deviations (recomended value = 3)")
NumStandardDeviations = float(input())

print("Enter Value for shearlet coefficients threshold, (recomended value = 1.2)")
shearletThresholdFactor = float(input())

print("Enter Value for number of frames to mean over, (recomended value = 20)")
framesToMean = int(input())

from shutil import copyfile

copyfile(fileName, "../OutputFiles/Denoised.brw")
copyfile("../OutputFiles/Denoised.brw", "../OutputFiles/DenoisedAndSheared.brw")
print("files prepared")


# In[3]:
	

dir1 = fileName
dir2 = '../OutputFiles/Denoised.brw'
dir3 = '../OutputFiles/DenoisedAndSheared.brw'


# In[4]:


# # function that loads specific channel from MEA device
def loadData(index):

     with h5py.File(dir1, 'r') as hdf:

        chsList = np.array(hdf.get('3BRecInfo/3BMeaStreams/Raw/Chs')[:])
        numChs = len(chsList)
        frames = np.array(hdf.get('3BRecInfo/3BRecVars/NRecFrames')[0])
        channelIdx = index
        ids = np.arange(channelIdx, frames * numChs, numChs)
        data = np.array(hdf.get('3BData/Raw'))[ids]
        # extract info relating to signal conversion 
        signalInversion = np.array(hdf.get('/3BRecInfo/3BRecVars/SignalInversion')[0])
        maxUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MaxVolt')[0]))
        minUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MinVolt')[0]))
        bitDepth = (np.array(hdf.get('/3BRecInfo/3BRecVars/BitDepth')[0]))
        sampmlingFrequency = (np.array(hdf.get('3BRecInfo/3BRecVars/SamplingRate')[0]))
        # convert MEA signal to have zero mean and correct scale        
        qLevel = 2**bitDepth 
        fromQLevelToUVolt = (maxUVolt - minUVolt) / qLevel
        singleElectrodeUV = data * signalInversion;
        singleElectrodeUV = singleElectrodeUV - (qLevel / 2);
        singleElectrodeUV = singleElectrodeUV * fromQLevelToUVolt
        return singleElectrodeUV, sampmlingFrequency, numChs, ids, frames


# # In[5]:


# # function that loads all of the data from MEA device
def loadDirectory(directory):

     with h5py.File(directory, 'r') as hdf:

        chsList = np.array(hdf.get('3BRecInfo/3BMeaStreams/Raw/Chs')[:])
        numChs = len(chsList)
        frames = np.array(hdf.get('3BRecInfo/3BRecVars/NRecFrames')[0])
        data = np.array(hdf.get('3BData/Raw'))

        # extract info relating to signal conversion 
        signalInversion = np.array(hdf.get('/3BRecInfo/3BRecVars/SignalInversion')[0])
        maxUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MaxVolt')[0]))
        minUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MinVolt')[0]))
        bitDepth = (np.array(hdf.get('/3BRecInfo/3BRecVars/BitDepth')[0]))
        sampmlingFrequency = (np.array(hdf.get('3BRecInfo/3BRecVars/SamplingRate')[0]))

        qLevel = 2**bitDepth 
        fromQLevelToUVolt = (maxUVolt - minUVolt) / qLevel
        singleElectrodeUV = data * signalInversion;
        singleElectrodeUV = singleElectrodeUV - (qLevel / 2);
        singleElectrodeUV = singleElectrodeUV * fromQLevelToUVolt
        return singleElectrodeUV, numChs, frames,sampmlingFrequency


# # In[6]:


# # function that loads specific settings of MEA device
def MEAsettings(directory):

     with h5py.File(directory, 'r') as hdf:
        signalInversion = np.array(hdf.get('/3BRecInfo/3BRecVars/SignalInversion')[0])
        maxUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MaxVolt')[0]))
        minUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MinVolt')[0]))
        bitDepth = (np.array(hdf.get('/3BRecInfo/3BRecVars/BitDepth')[0]))
        qLevel = 2**bitDepth 
        fromQLevelToUVolt = (maxUVolt - minUVolt) / qLevel
        return  fromQLevelToUVolt,  qLevel, signalInversion


# # In[7]:


# # Denoising algorithm consisting of Continious Wavelt Transform and PCA
# # Takes in channel data
def BinaryClssification(singleElectrodeUV):
    # CWT
    y = singleElectrodeUV
    coef, freqs=pywt.cwt(y,np.arange(1,50,2),'morl')
    d = coef
    # transpose continious wavelet transform matrix to be fed into PCA
    d = d.transpose()
    # PCA
    pca = PCA(n_components = 2)
    pca.fit(d)
    Z = pca.transform(d)
    # reshape the cluster to be sperical rather than eliptical, easyer to do standard deviation calculations.     
    eigenvalues = pca.explained_variance_
    Z[:,0] /= eigenvalues[0]**(1/2)
    Z[:,1] /= eigenvalues[1]**(1/2)
    # K - Mean with number of cluster set to 1 to determine center of the cluster
    kmeans = KMeans(n_clusters=1).fit(Z)
    # Find the coordinates of the center of the cluster and distances of all points to that center    
    centroids = kmeans.cluster_centers_
    Distance = cdist(Z, centroids)
    # calculate standard deviation on all distances
    satandardDiv = np.std(Z)
    #  create a mask where all the points that are outside of X * sigma are kept and other are disgarded   
    threshold = NumStandardDeviations * satandardDiv
    mask = map(lambda x : 1 if (x > threshold) else 0, Distance)
    mask = list(mask)
    #  Multiply the mask by the original signal and return the result   
    re = singleElectrodeUV * mask
    return re



# # In[8]:


# # shearlet transform function, takes in 2d array (a frame)
def shearlets(matrix):
    #  hyperparameter settings (including level at which to threshold)   
    sigma = 30
    scales = 2
    thresholdingFactor = shearletThresholdFactor
    # Converting 2D matrix into flaot    
    matrix = matrix.astype(float)
    X = matrix 
    ## create shearlets system
    shearletSystem = pyshearlab.SLgetShearletSystem2D(0,X.shape[0], X.shape[1], scales)
    # decomposition, produces shearlet coefficients
    coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)
    # calculating Root Mean Square value of each coeficient vector that is assosiated with a pixel
    # i.e for sherlet system made out of scales = 2, produces coeficient vector of length 17 for each pixel
    # RMS value is worked out on each of these vectors and the multiplied by the vector 
    # A 1xnShearlets array containing the root mean squares (L2-norm divided by
    # sqrt(X*Y)) of all shearlets stored in shearletSystem["shearlets"]. These values can be used to normalize
    # shearlet coefficients to make them comparable.
    oldCoeffs = coeffs.copy()
    weights = np.ones(coeffs.shape)
    for j in range(len(shearletSystem["RMS"])):
        weights[:,:,j] = shearletSystem["RMS"][j]*np.ones((X.shape[0], X.shape[1]))
    #  Thresholding the coefficients based on the setting in the hyperparameters and RMS weights 
    #  Setting coefficients to 0 for value that do not pass the threshold
    coeffs = np.real(coeffs)
    zero_indices = np.abs(coeffs) / (thresholdingFactor * weights * sigma) < 1
    coeffs[zero_indices] = 0
    # reconstruction of the signal thresholded coefficients, returning the reconsturcted signal
    Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)
    return Xrec



# # In[9]:


singleElectrodeUV, sampmlingFrequency, numChs, ids, frames = loadData(0) 
numberOfSmples = frames * numChs
VectorOfOne = np.zeros(numberOfSmples)


def do_32(i):
    # if(i%50 ==0):
    print(i," / 4096   analysed")
    # load data channel
    singleElectrodeUV, sampmlingFrequency, numChs, ids, frames = loadData(i)
    FilteredsingleElectrodeUV =  np.where(singleElectrodeUV > 3000,0, singleElectrodeUV)
    # checking if channel is empty and if not perform binary classification
    Max = np.max(FilteredsingleElectrodeUV)
    if(Max == 0.0):
        print("///////////////////////////////////////////")
        return FilteredsingleElectrodeUV
    else:
        # result = BinaryClssification(singleElectrodeUV)
        result = BinaryClssification(FilteredsingleElectrodeUV)
        return result


# multiprocessing new way
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool(processes =4)
#     data = p.map(do_32,range(64*64))
    data= p.map(do_32,range(64*64))
    p.close()

vData = np.vstack(data)
rt = np.transpose(vData)
f = rt.flatten()


# take the processed binary classification of all the channels and prepare it to be written back into the
#  brw format
fromQLevelToUVolt,  qLevel, signalInversion = MEAsettings(dir1)
f = f / fromQLevelToUVolt
f = f + (qLevel / 2);
f = f / signalInversion;


final_array = np.array(f, dtype='int16')


# append a copy of dir1 with the new results.
with h5py.File(dir2, mode = 'a') as f:
    print(list(f.items()))
    g1 = f.get('3BData')
 
    #delete old version of 'Raw'
    delete = 'Raw'
    if delete in g1.keys():
        del g1['Raw']
        
    #create new version of 'Raw' 
    dataset = "Raw"
    if dataset not in g1.keys():
        print("dataset is not here yet!")
        g1.create_dataset('Raw', data = final_array, dtype='uint16') #dtype='int16'


    #display keys to make sure 'Raw' file exists 
    g1_items = list(g1.items())
    print('items in group 1: ', g1_items)
    print(g1.keys())


xstr = (53,20)
# Load data
with h5py.File(dir2, 'r') as hdf:
    
    chsList = np.array(hdf.get('3BRecInfo/3BMeaStreams/Raw/Chs')[:])
    numChs = len(chsList)
    frames = np.array(hdf.get('3BRecInfo/3BRecVars/NRecFrames')[0])
    selectedChannel = (chsList['Row'] == int(xstr[0])) * (chsList['Col'] == int(xstr[1]))
    channelIdx = np.where(selectedChannel == True)[0][0]
    ids = np.arange(channelIdx, frames * numChs, numChs)
    data = np.array(hdf.get('3BData/Raw'))[ids]
    # extract info relating to signal conversion 
    signalInversion = np.array(hdf.get('/3BRecInfo/3BRecVars/SignalInversion')[0])
    maxUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MaxVolt')[0]))
    minUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MinVolt')[0]))
    bitDepth = (np.array(hdf.get('/3BRecInfo/3BRecVars/BitDepth')[0]))
    sampmlingFrequency = (np.array(hdf.get('3BRecInfo/3BRecVars/SamplingRate')[0]))
    
qLevel = 2**bitDepth 
fromQLevelToUVolt = (maxUVolt - minUVolt) / qLevel
singleElectrodeUV = data * signalInversion;
singleElectrodeUV = singleElectrodeUV - (qLevel / 2);
singleElectrodeUV = singleElectrodeUV * fromQLevelToUVolt


# plot the channel
# plt.figure(figsize=(25,7.5))
# plt.plot(range(len(singleElectrodeUV))/sampmlingFrequency, singleElectrodeUV) 
# plt.title('singleElectrodeUV')
# plt.show()

print('denoising completed, initialising shearlets ...')

rawData, numChs, frames, sampmlingFrequency =  loadDirectory(dir2)


# Take the mean of X number of frames, windowSize indicates number of frames to take 
OutData = []
windowSize = framesToMean
for pos in range(frames - windowSize):
    if(pos%1000 == 0):
        print("pos: ",pos,end="\r", flush=True)
    storeArr = []
    start = pos
    
    for i in range(start,start + windowSize):
        frame = i
        data = rawData[numChs * frame :numChs * (frame + 1)]
        storeArr.append(data) 

    storeArr = np.array(storeArr)
    meanWindow = storeArr.mean(axis=0)
    OutData.append(meanWindow)



# Code that takes mean frames and feeds it into shearlet function 
def shear(i):
    I = OutData[i].reshape(64,64)
    Xrec = shearlets(matrix = I)
    # Flatten frames back into array 
    XrecArr =  Xrec.flatten()
    p = int((i / len(OutData))*100)
    print("percent channels analysed: ",p)
    return XrecArr

# multiprocessing new way
from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool(processes =4)
#     data = p.map(do_32,range(64*64))
    shearletMatrix= p.map(shear,range(len(OutData)))
    p.close()


# Code that takes result of shearing function (frames) and converts it back into 1D signal
# to be written into a new file latter
a = np.array(shearletMatrix)
f =  a.flatten()
d =  len(rawData) -  len(f)
ff = f.tolist()
rr = rawData[-d:].tolist()
n = ff + rr
out = np.array(n)



# multilying the result by some number, this is because shealet function tends to shrink signal
# amplitude

out = np.multiply(out,5)

# mask = map(lambda x : 1 if (abs(x) > 0) else 0, out)
# mask = list(mask)
# r = rawData * mask


# load settings data from MEA device, to be used to write the results back into .brw file 
print('preparing to write to new file ...')
fromQLevelToUVolt,  qLevel, signalInversion = MEAsettings(dir2)


# reverse back to raw signal format to be written into the software
out = out / fromQLevelToUVolt
out = out + (qLevel / 2);
out = out / signalInversion;


final_array = np.array(out, dtype='int16')
print(final_array[0:10])


# Write back into a brw file
print('writing file ...')
with h5py.File(dir3, mode = 'a') as f:
    # print(list(f.items()))
    g1 = f.get('3BData')
        
    #delete old version of 'Raw'
    delete = 'Raw'
    if delete in g1.keys():
        del g1['Raw']
        
    #create new version of 'Raw' 
    dataset = "Raw"
    if dataset not in g1.keys():
        # print("dataset is not here yet!")
        g1.create_dataset('Raw', data = final_array, dtype='uint16') #dtype='int16'


    #display keys to make sure 'Raw' file exists 
    g1_items = list(g1.items())
    # print('items in group 1: ', g1_items)
    print(g1.keys())


# display any channel in the new denoised file
xstr = (53,20)
# Load data
with h5py.File(dir3, 'r') as hdf:
    chsList = np.array(hdf.get('3BRecInfo/3BMeaStreams/Raw/Chs')[:])
    numChs = len(chsList)
    frames = np.array(hdf.get('3BRecInfo/3BRecVars/NRecFrames')[0])
    selectedChannel = (chsList['Row'] == int(xstr[0])) * (chsList['Col'] == int(xstr[1]))
    channelIdx = np.where(selectedChannel == True)[0][0]
    ids = np.arange(channelIdx, frames * numChs, numChs)
    data = np.array(hdf.get('3BData/Raw'))[ids]
    # extract info relating to signal conversion 
    signalInversion = np.array(hdf.get('/3BRecInfo/3BRecVars/SignalInversion')[0])
    maxUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MaxVolt')[0]))
    minUVolt = (np.array(hdf.get('/3BRecInfo/3BRecVars/MinVolt')[0]))
    bitDepth = (np.array(hdf.get('/3BRecInfo/3BRecVars/BitDepth')[0]))
    sampmlingFrequency = (np.array(hdf.get('3BRecInfo/3BRecVars/SamplingRate')[0]))
    
qLevel = 2**bitDepth 
fromQLevelToUVolt = (maxUVolt - minUVolt) / qLevel
singleElectrodeUV = data * signalInversion;
singleElectrodeUV = singleElectrodeUV - (qLevel / 2);
singleElectrodeUV = singleElectrodeUV * fromQLevelToUVolt

# visualize result from randomly selected channel 
print('Code finished running, denoising + shearlet application completed')
plt.figure(figsize=(25,7.5))
plt.plot(range(len(singleElectrodeUV))/sampmlingFrequency, singleElectrodeUV) 
plt.title('filtering signal')
plt.show()






