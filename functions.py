import ogr, osr
import numpy as np
import geopandas
import matplotlib.pyplot as plt
from PIL import Image

from affine import Affine
def shiftImage(naip_meta, Xoffset, Yoffset):
    naip_metaNew = naip_meta#.copy() #it keeps the old one
    naip_metaT = naip_metaNew['transform']
    naip_metaNew['transform'] = Affine(naip_metaT[0], naip_metaT[1], naip_metaT[2]+Xoffset , \
                           naip_metaT[3], naip_metaT[4], naip_metaT[5]+Yoffset)
    return naip_metaNew


def transformGPS_coord(pointXY, inputEPSG = 32632, outputEPSG = 4326 ):
    pointX = pointXY[1]
    pointY = pointXY[0]
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(pointX, pointY)    
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)    
    # transform point
    point.Transform(coordTransform)    
    # print point in EPSG 4326
    return (point.GetX(), point.GetY())


#Test if a point coordinate (63.2...; 5.1...) is in the map: naip_meta is the geoTiff header info
def isPointOnMap(point,  naip_meta, radius=0):
    naip_transform = naip_meta["transform"]
    pointX, pointY = transformGPS_coord(point, outputEPSG = 32632, inputEPSG = 4326 )
    u = int((pointX - naip_transform[2]) / naip_transform[0])
    v = int((pointY - naip_transform[5]) / naip_transform[4])

    if u >= radius and u <= naip_meta["width"]-radius and v >= radius and v <= naip_meta["height"]-radius:
       return True
    else:
        return False
    

# Extract window of given radius from map at a certain location
def extractWindow(point, Map, radius, naip_meta):

    if isPointOnMap(point, naip_meta, radius=radius):
        naip_transform = naip_meta["transform"]
        pointX, pointY = transformGPS_coord(point, outputEPSG = 32632, inputEPSG = 4326 )
        xOffset = int((pointX - naip_transform[2]) / naip_transform[0])
        yOffset = int((pointY - naip_transform[5]) / naip_transform[4])        
        imageSample = Map[:, yOffset-radius:yOffset+radius, xOffset-radius:xOffset+radius].astype('float32')
        #if imageSample is all 0: no data
        if np.sum(imageSample) == 0:
            imageSample = np.array([-1])
        else:
            pass
    else:
        imageSample = np.array([-1])
    
    return imageSample

# If window==(-1) it means the acquisition was not OK (because for example point out of map or too close to the border) 
def isWindowOK(window):
    if len(window) == 1: 
        return False
    else:
        return True
    
def isLidarOK(window):
    if window.max() == 0 or len(window) == 1:
        return False
    else:
        return True


#Show a window
def showWindow(window):
    if isWindowOK(window): 
        if window.shape[0] == 8: #WV2
            imageRGB = (np.einsum('kij->ijk', np.take(window, [4,2,1], axis = 0))*255).astype('uint8')
        else: #Pleiades
            imageRGB = (np.einsum('kij->ijk', np.take(window, [0,1,2], axis = 0))*255).astype('uint8')
        # plt.figure()
        plt.imshow(imageRGB)
        # im = Image.fromarray(imageRGB).show()
    else:        
        print("-- Image = none")
 
       
def saveWindow_asBMP(filename, window):
    if window.shape[0]==8: #WV2
        imageRGB = (np.einsum('kij->ijk', np.take(window, [4,2,1], axis=0))*255).astype('uint8')
    if window.shape[0]==4: #Pleiades
        imageRGB = (np.einsum('kij->ijk', np.take(window, [0,1,2], axis=0))*255).astype('uint8')
    im = Image.fromarray(imageRGB)
    im.save(filename, 'BMP')
        


#Compute NDVI
def computeNDVI (image):
    if image.shape[0] == 8: #multi-channel WorldView2 image 
        NIR = image[6,:,:]
        RED = image[4,:,:]
    elif image.shape[0] == 4: #multi-channel Pleiades image
        NIR = image[3,:,:]
        RED = image[0,:,:]  
    else:
        print("Image with not enough channels or wrong image")
        return None         
    #prevent division by zero
    # den = NIR + RED
    # den[np.where(den==0)] = 1
    
    NDVI = ( (NIR - RED) / (NIR + RED) ).astype('float32')
    # NDVI[NDVI==np.nan] = 0
    return NDVI




# Create a Python list of poles location from .SHX file   
def scanPoles(fileName):
    shapefile = geopandas.read_file(fileName)       
    coordinatesPoles = np.zeros( (len(shapefile), 2) )
    for i in range(shapefile.index.start, shapefile.index.stop):
        s = str(shapefile.loc[i] ['geometry'])
        s = s[s.find('(')+1:s.find(')')].split(' ')[0:2]
        coordinatesPoles[i,0] = float(s[0]) 
        coordinatesPoles[i,1] = float(s[1])          
    return coordinatesPoles


#Interpolate power lines (DEPRECATED) 
import math
def scanLines(filename, coverage):
    shapefile = geopandas.read_file(filename) 
    coordinatesPoles2 = []
    for i in range(shapefile.index.start, shapefile.index.stop):
        s = str(shapefile.loc[i] ['geometry'])
        s = s[s.find('(')+1:s.find(')')].split(' ')
        index = 0
        tmp = np.zeros( (40, 2) )
        for k in range(0, len(s),3 ):
            tmp[index,0] = float(s[k])
            tmp[index,1] = float(s[k+1])
            index = index + 1
        tmp = tmp[0:index,:] 
        coordinatesPoles2.append(tmp)
        
    Location = np.zeros( (10000, 2))    
    # Interpolation
    index = 0
    for group in range(0, len(coordinatesPoles2) ):
        M = len(coordinatesPoles2[group])
        for sample in range(0, M-1):
    
            X1 = coordinatesPoles2[group][sample, 0]
            Y1 = coordinatesPoles2[group][sample, 1]
            X2 = coordinatesPoles2[group][sample+1, 0]
            Y2 = coordinatesPoles2[group][sample+1, 1]
            x1, y1 = transformGPS_coord(np.array([X1, Y1]), outputEPSG = 32632, inputEPSG = 4326 ) 
            x2, y2 = transformGPS_coord(np.array([X2, Y2]), outputEPSG = 32632, inputEPSG = 4326 ) 
            distance = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
            N = math.ceil(distance / coverage)
            if N>1:
                for kk in range(1, N):
                    Px = x1+ (kk*(x2-x1)/N)
                    Py = y1 + ((y2-y1)/(x2-x1))*(Px-x1)
                    Py, Px = transformGPS_coord(np.array([Py, Px]), inputEPSG = 32632, outputEPSG = 4326 ) 
                    Location[index,0] = Px
                    Location[index,1] = Py
                    index = index + 1
    
     
    Location = Location[0:index-1,:]
    return Location


#Interpolate power lines: 
import re
def scanLines2(filename, coverage, output_format='4326'):
    
    shapefile = geopandas.read_file(filename)
    
    def computeDistance(A,B):
        distance = np.sqrt( (A[0] - B[0] )**2 + (A[1] - B[1] )**2 ) 
        return distance
    
    location = []
    for i in range(0, len(shapefile)):
        segment = str(shapefile.loc[i] ['geometry'])
        tmp=re.findall("\d+\.\d+", segment)   #A=re.findall("\d+\.\d+", s) or re.findall(r"[-+]?\d*\.\d+|\d+", s)
        vertices = np.empty((int(len(tmp)/2), 2))
        for i in range(0, len(vertices)):
            vertices[i,0], vertices[i,1] = transformGPS_coord( (float(tmp[2*i]), float(tmp[2*i+1])) , outputEPSG = 32632, inputEPSG = 4326 ) 
       
        for i in range(0, len(vertices)-1):
            A = vertices[i,:]
            B = vertices[i+1,:]
            location.append(A)
            distance = computeDistance(A,B)
            if distance < coverage: #if distance<coverage: no need to interpolate
                pass
            else: #compute intermediate points by interpolation
                #how many points needed?
                N = int(np.ceil(distance / coverage))
                for kk in range(1, N): #linear interpolation
                    m = (B[1]-A[1])/(B[0]-A[0])
                    Px = A[0] + (kk*(B[0]-A[0])/N)
                    Py = A[1] + m*(Px-A[0])
                    location.append(np.array([Px, Py]))
                    
        location.append(vertices[-1,:]) #include the last vertex of the segment
    
    location = np.asarray(location)
    
    if output_format == '32632':
        pass
    elif output_format == '4326':
        # if we want the output to be in 4326 geo-reference (longitude, latitude)
        for i in range(0, len(location)):
            Px = location[i,0]; Py = location[i,1]; 
            [lat, long] = transformGPS_coord( (Py, Px), inputEPSG = 32632, outputEPSG = 4326 )
            location[i,:] = [long,lat]
    else:
        print("Wrong output_format:  32632 or 4326")
    return location


#------------------------------------------------------------------------------
#LIDAR-------------------------------------------------------------------------
import rasterio

def LIDAR_organizeLocations(locations, folder): 
    """ Create a list in which each element is an array of [locations]
    clusterLoc[0]: all locations along the lines that are covered by Cluster1 
    clusterLoc[1]: all locations along the lines that are covered by Cluster2
    ...
    counter: number of suitable locations (locations covered by LiDAR data)
    """    
    def LIDAR_findCluster(location, path): 
        """Given a location, find out which cluster contains that location"""
        cluster = 0
        pointX, pointY = transformGPS_coord(location, outputEPSG = 32632, inputEPSG = 4326 )
        for i in range(1,7):
            naip_meta = rasterio.open(path+'/Cluster '+str(i)+'.tif').profile
            naip_transform = naip_meta['transform']
            u = int((pointX - naip_transform[2]) / naip_transform[0])
            v = int((pointY - naip_transform[5]) / naip_transform[4])
            if u > 0 and v > 0 and u < naip_meta['width'] and v < naip_meta['height']:
                cluster = i
                break
        return cluster

    cluster_number = np.zeros((len(locations), 1))
    for index in range(0, len(locations)):
        loc = locations[index]
        # --which cluster to consider
        cluster_number[index] = LIDAR_findCluster(loc, folder)
    locations = np.hstack((locations, cluster_number))
    locations = locations[locations[:,2].argsort()]
    counter = np.where(locations[:,2]>=1)[0].shape[0] #How many points there are: the ones from Cluster 1 to 6
    clusterLoc = []
    for i in range(1, 7): #from Cluster1 to Cluster6
        clusterLoc.append(locations[np.where(locations[:,2]==i)][:,0:2])
    return clusterLoc, counter


def LIDAR_extractWindow(point, Lidar_map, radius, naip_meta):    
    if isPointOnMap(point, naip_meta, radius=radius):
        naip_transform = naip_meta['transform']
        pointX, pointY = transformGPS_coord(point, outputEPSG = 32632, inputEPSG = 4326 )
        xOffset = int((pointX - naip_transform[2]) / naip_transform[0])
        yOffset = int((pointY - naip_transform[5]) / naip_transform[4]) 
        if len(Lidar_map.shape) == 2:
            window = Lidar_map[yOffset-radius:yOffset+radius, xOffset-radius:xOffset+radius].astype('uint8')
        else:
            window = Lidar_map[:, yOffset-radius:yOffset+radius, xOffset-radius:xOffset+radius].astype('uint8')
                    
    else:
        window = np.array([-1])

    return window



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def split_image(image, new_radius): #image.shape: [Nchannels, radius, radius]   
    L = new_radius*2
    dataset = np.empty(( int((image.shape[1]/L)**2), image.shape[0], L, L ))
    k = 0
    for row in range(0, image.shape[1], L):
        for col in range(0, image.shape[2], L):    
            dataset[k,:,:,:] = image[:, row:row+L, col:col+L]
            k+=1
    return dataset

def split_mask(mask, new_radius): #mask.shape: [radius, radius]   
    L = new_radius*2
    dataset = np.empty(( int((mask.shape[0]/L)**2), L, L ))
    k = 0
    for row in range(0, mask.shape[0], L):
        for col in range(0, mask.shape[1], L):    
            dataset[k,:,:] = mask[row:row+L, col:col+L]
            k+=1
    return dataset
            

def export_GEOtiff(filename, Map, naip_meta):     
    naip_meta['dtype'] = Map.dtype
    if len(Map.shape)==2:
        Map = np.expand_dims(Map, axis=0)
    naip_meta['count'] = Map.shape[0]    
    assert naip_meta['count'] <=8
    with rasterio.open(filename, 'w', **naip_meta) as dst:
        dst.write(Map)
        
def createQGISMap(filename, patches, locations, meta_header, alphaChannel = False, format_loc = None,):
    """Map back to QGIS an array of patches [N,H,W], each centered at loc defind in locations"""
    assert len(patches) == len(locations)
    
    if alphaChannel:
        changeMap = np.zeros(( 2, meta_header['height'], meta_header['width'])).astype(patches.dtype)
    else:
        changeMap = np.zeros((meta_header['height'], meta_header['width'])).astype(patches.dtype)
    
    h = int(patches.shape[1]/2)
    T = meta_header['transform']
    for i in range(0,len(locations)):
        if format_loc == '4326': #locations = (long, lat)
            #Trasform into linear crs: (295089, 6806353)
            locations[i,:] = transformGPS_coord(locations[i,:], outputEPSG = 32632, inputEPSG = 4326 )
        v = int( (locations[i,0] - T[2] ) / T[0] )
        u = int( (locations[i,1] - T[5] ) / T[4] )        
        
        if alphaChannel:
            changeMap[0, u-h:u+h, v-h:v+h] = patches[i,:,:]
            changeMap[1, u-h:u+h, v-h:v+h] = 255
        else:
            changeMap[u-h:u+h, v-h:v+h] = patches[i,:,:]
        
    export_GEOtiff(filename, changeMap, meta_header)
    
    
    
        
        
def normalizeArray(array, new_range):     
    m = (new_range[1]-new_range[0])/(array.max() - array.min())
    array = m*(array - array.max()) + new_range[1]
    return array

    
    
    
    
class tilesGenerator():
    def __init__(self, image, stepSize):
        #parameters defined by user
        self.im = image
        self.T = stepSize
        
        #internal parameters       
        self.n_tiles_per_row = int(np.floor(self.im.shape[1]/self.T))
        self.n_tiles_per_col = int(np.floor(self.im.shape[2]/self.T))        
        self.indeces = np.empty((2, self.n_tiles_per_row * self.n_tiles_per_col)).astype(int)
        
        #---Storing the tiles is too much memory-consumping---
        # self.tiles = np.empty((self.n_tiles_per_row * self.n_tiles_per_col, self.im.shape[0], self.T, self.T)).astype(int)

        
    def image2tiles(self):
        #Create an array of tiles and an array with indeces of each tile in the initial big image
        k = 0       
        for row in range(0, self.T*self.n_tiles_per_row, self.T):
            for col in range(0, self.T*self.n_tiles_per_col, self.T):
                self.indeces[:, k] = row, col
                # window = self.im[:, row:row+self.T, col:col+self.T]                
                # self.tiles[k,:,:,:] = window
                k = k+1
        
    # def select_tiles(self):
    #     #Select only tiles inside the map, so not all-black tiles