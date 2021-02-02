import matplotlib.pyplot as plt
import numpy as np
import rasterio
import sys
import time
from PIL import ImageEnhance, Image
import os
import gdal
import tkinter as tk    # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

# if not ('functions_loaded' in locals()):
#     root = tk.Tk()
#     root.withdraw()
#     print(">>Select folder containing 'functions.py': (minimize window) ")
#     folder_selected = tk.filedialog.askdirectory()
#     functions_loaded = True

# sys.path.append(folder_selected) #location of functions_ver2



import functions as f
tk.Tk().withdraw() 
print(">>Select satellite image: (minimize window)")
# filename = askopenfilename() 
filename = 'D:/Master/Processing/GeoTiffs/9-16-2017_Ortho_4Band.tif'
print("---Loading Map---")
dataset = rasterio.open(filename)
naip_meta = dataset.profile
naip_meta = f.shiftImage(naip_meta, 4, 0)
Map = dataset.read([1,2,3,4])
dataset.close()
cutOff = [450, 550, 680, 2370] 
print("---Loading Map: end---")


print(">>Select LiDAR map: (minimize window)")
filename = 'D:/Master/Processing/GeoTiffs/SpeciesAskvoll.tif'
dataset = rasterio.open(filename)
naip_meta_LIDAR = dataset.profile
LIDAR_map = dataset.read([1])[0,:,:]
dataset.close() 



"""
print(">>Select .shp power poles cordinates: (minimize window)")
filename = 'D:\Master\Processing\Kraftnett2_bigger\Kraftnett_Mast.shp'
polesCoordinate = f.scanPoles(filename)
 

print(">>Select .shp power line cordinates: (minimize window)")
filename = 'D:\Master\Processing\Kraftnett2_bigger\Kraftnett_Kraftlinje.shp'
locations = f.scanLines2(filename, coverage=10)

rows = 10000
cols = 2
counter = 0
lat = 5.0750
long = 61.3250
locations = np.zeros((rows, cols), dtype=float)

for i in range(rows):
    locations[i,:] = [lat, long]
    long += 0.0005
    counter += 1
    if counter == 100:
        long = 61.3250
        lat += 0.0005
        counter = 0
        

"""
N=11000

def long_gen():
    return round(np.random.uniform(5.0600, 5.2700), 8)
def lat_gen():
    return round(np.random.uniform(61.3200, 61.3800), 8)


radius=128;  kk = 0
SATimages_dataset = np.empty((N, 4, 2*radius, 2*radius)) #(Nimages, Nchannels, height, width)
LIDARimages_dataset = np.empty((N, 2*radius, 2*radius)) #(Nimages, Nchannels, height, width)

for i in range(0, N):
    lat = lat_gen()
    long = long_gen()
    window_SAT = f.extractWindow([long, lat], Map, radius, naip_meta, cutOff)
    window_LIDAR = f.LIDAR_extractWindow([long, lat], LIDAR_map, radius, naip_meta_LIDAR)
    
    if f.isWindowOK(window_SAT) and f.isLidarOK(window_LIDAR):
        SATimages_dataset[kk,:,:,:] = window_SAT[0:4,:,:] #only RGB
        LIDARimages_dataset[kk,:,:] = window_LIDAR
        kk = kk + 1
        
SATimages_dataset = SATimages_dataset[0:kk,:,:,:].astype('float32')
LIDARimages_dataset = LIDARimages_dataset[0:kk,:,:].astype('uint8')

print(SATimages_dataset.shape)
print(LIDARimages_dataset.shape)
np.save('Datasets/SAT_species_256revised.npy', SATimages_dataset)
np.save('Datasets/species_256revisedraw.npy', LIDARimages_dataset)
