# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:05:15 2021

@author: sindr
"""
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import functions as f
import pickle

def pred_gen(preds,dim=256):
    mask = np.zeros((dim,dim,1)).astype('uint8')
    high_mask = np.zeros((dim,dim,1))
    for z in range (6):
        for x in range(dim):
            for y in range(dim):
                if preds[0,x,y,z] > high_mask[x,y]:
                    mask[x,y] = z
                    high_mask[x,y] = preds[0,x,y,z]
    return mask


filename = 'D:/Master/Processing/GeoTiffs/9-16-2017_Ortho_4Band.tif'
SAT = rasterio.open(filename)
SAT_header = SAT.profile
SAT_map = SAT.read([1,2,3,4]) #RGB-NDVI
SAT.close()

filename = 'D:/Master/Processing/GeoTiffs/SpeciesAskvoll.tif'
NIBIO = rasterio.open(filename)
meta_header = NIBIO.profile
NIBIO_map = NIBIO.read([1]).astype('uint8')
NIBIO.close()


        
stepSize= 256
tiles = f.tilesGenerator(SAT_map, stepSize)
tiles.image2tiles()
indeces = tiles.indeces
            

"""Output: same shape as NIBIO_map"""
output_map = np.zeros(NIBIO_map.shape).astype(NIBIO_map.dtype)


model = pickle.load(open('Models/Species256AskvollAtt.sav', 'rb'))


for i in range(0, indeces.shape[1]):
    #Extract window from that location
    row = indeces[0,i]; col = indeces[1,i]
    window = SAT_map[:, row:row+stepSize, col:col+stepSize]
    #Select only tiles inside the map, so not all-black tiles
    if np.sum(window) == 0:
        predicted_output = window[0:1,:,:]*0 
        print("empty window", i)
    else:
        window = window.transpose((1,2,0))
        window = np.expand_dims(window, axis=0)
        predicted_output = model.predict(window)
        predicted_output = pred_gen(predicted_output)
        predicted_output = predicted_output.transpose((2,0,1))
        # predicted_output = np.argmax(predicted_output, axis=3)
        # print(predicted_output.mean())
        # print("predicted window", i)
    # put back the predicted output
    output_map[:,row:row+stepSize, col:col+stepSize] = predicted_output

f.export_GEOtiff('output_map.tif', output_map, meta_header)