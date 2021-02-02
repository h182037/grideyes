# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:12:16 2020

@author: sindr

0: No Trees
1: Gran (Spruce)
2: Furu (Pine)
3: Bar (Coniferous)
4: Blanding (Mixed)
5: Lauv (Decidous)
"""
import segmentation_models as sm
import numpy as np
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle
import model_lib as ml




dim = 256
classes = 6
species = np.load("Datasets/species_256revisedcat.npy")
sat = np.load("Datasets/SAT_species_256revised.npy")
sat = sat.transpose((0,2,3,1))
# species = tf.keras.utils.to_categorical(species, None)
# np.save('species_256revisedcar.npy', species)

print(species.shape)
print(sat.shape)
# sat = sat[:,:,:,:]

thresh= 3000
test_spec = species[thresh:]
test_sat = sat[thresh:]
species = species[:thresh]
sat = sat[:thresh]

model = pickle.load(open('Models/Species256AskvollAtt.sav', 'rb'))


# model = sm.FPN('vgg16', input_shape=(dim,dim,4), classes=6, encoder_weights=None)
# model = ml.get_UnetPP(num_classes=6, shape=(256,256,4), number_of_filters=1)
# model = ml.get_Unet(classes=6, shape=(256,256,4))
# model = ml.get_rec_unet(shape=(256,256,4), depth=3, features=32, data_format='channels_last', classes=6)
# model = ml.get_ResUnet(input_size=(256,256,4))
# model = ml.get_resFPN(input_shape=(256,256,4), nb_labels=6, pyramid=256, width=4)
# model = ml.get_att_unet(shape=(256,256,4), classes=6, dropout=0.2, depth=8, features=8)


# AttNet: bs: 16, lr: 1e-3, dropout: 0.2, depth: 8, features: 8  | cc: 0.1741, focal: 0.0242, jaccard: 0.8782, dice: 0.8116
# AttNet: bs: 16, lr: 1e-3, dropout: 0.2, depth: 4, features: 32 | cc: 0.1186, focal: 0.0165, jaccard: 0.7780, dice: 0.6937
# AttNet: bs: 16, lr: 1e-3, dropout: 0.2, depth: 2, features: 64 | cc: 0.1248, focal: 0.0185, jaccard: 0.7892, dice: 0.7068
# AttNet: bs: 16, lr: 1e-5, dropout: 0, depth: 2, features: 64 | cc: 0.1950, focal: 0.0254, jaccard: 0.8993, dice: 0.8355
# AttNet: bs: 16, lr: 1e-3, dropout: 0, depth: 2, features: 64      | cc: 0.1271, focal: 0.0194, jaccard: 0.7856, dice: 0.7030
# FPNRes: bs 16, lr: 1e-3, pyramid: 256, width: 4  |                | cc: 0.3493, focal: 0.0442, jaccard: 0.7566, dice: 0.6918
# UnetResRec: bs: 8, lr: 3e-5, depth: 3, features: 32               | cc: 0.1439, focal: 0.0229, jaccard: 0.8184, dice: 0.7486
# UnetPP: bs: 16, lr: 3e-5 |    |   |     |    |    |               | cc: 0.1552, focal: 0.0247, jaccard: 0.8170, dice: 0.7381
# Unet: bs 16, lr 3e-5, dropout=None |    |    |    |               | cc: 0.1081, focal: 0.0190, jaccard: 0.7470, dice: 0.6634
# Unet: bs: 16, lr: 3e-5, dropout: 0.1 |  |    |    |               | cc: 0.1224, focal: 0.0180, jaccard: 0.7750, dice: 0.6902
epochs = 500
batch = 16
lr = 1e-3 



# opt = keras.optimizers.Adam(learning_rate=lr,
#                                 decay=lr/epochs
#                                 )
# model.compile(
#     opt,
#     sm.losses.categorical_crossentropy,
#     metrics=[ sm.losses.CategoricalFocalLoss(),sm.losses.JaccardLoss(), sm.losses.DiceLoss()]

# )
# model.summary()


# my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, verbose=1)]
# model.fit(x=sat, y=species, batch_size=batch, epochs=epochs, validation_split=0.1, callbacks=my_callbacks)
def setConf(conf, data):
    val_preds = model.predict(data)
    # val_preds = (val_preds > conf)
    return val_preds

val_preds = setConf(0.6, sat)

def displayResult(index, cl, conf=None):
    if conf:
        setConf(conf)
    plt.imshow(sat[index,:,:,0:3], interpolation='none')
    plt.show()
    
    plt.imshow(species[index,:,:,cl], interpolation='none')
    plt.show()
    
    plt.imshow(val_preds[index,:,:,cl], interpolation='none')
    plt.show()
    
test_preds = setConf(0.6, test_sat)

def validate(index, cl, conf=None):
    if conf:
        setConf(conf, test_sat)
    plt.imshow(test_sat[index,:,:,0:3], interpolation='none')
    plt.show()
    
    plt.imshow(test_spec[index,:,:,cl], interpolation='none')
    plt.show()
    
    plt.imshow(test_preds[index,:,:,cl], interpolation='none')
    plt.show()
# displayResult(10,0, 0.2)

mask_preds = model.predict(test_sat)


def showMask(index, conf=0.3, alpha=0.5):
    plt.imshow(test_sat[index,:,:,0:3], interpolation='none')
    plt.show()
    mask_test = np.zeros((dim,dim,1))
    for z in range (6):
        for x in range(dim):
            for y in range(dim):
                if test_spec[index,x,y,z] == 1:
                    mask_test[x,y] = z
    plt.imshow(mask_test, cmap = 'Greens', vmin=0, vmax=5, interpolation='none')
    plt.show()
    mask = np.zeros((dim,dim,1)).astype('uint8')
    high_mask = np.zeros((dim,dim,1))
    for z in range (6):
        for x in range(dim):
            for y in range(dim):
                if mask_preds[index,x,y,z] > high_mask[x,y]:
                    mask[x,y] = z
                    high_mask[x,y] = mask_preds[index,x,y,z]
    plt.imshow(mask, cmap = 'Greens', vmin=0, vmax=5, interpolation='none')
    plt.show()
    plt.imshow(test_sat[index,:,:,0:3], interpolation='none')
    plt.imshow(mask, cmap = 'Greens', vmin=0, vmax=5, interpolation='none', alpha=alpha)
    plt.show()
    


# pickle.dump(model, open('Models/Species256AskvollAtt.sav', 'wb'))

# for x in range((len(lidar))):
#     for i in range(dim):
#         for j in range(dim):
#             point = lidar[x,i,j]
#             if point > 0 and point < 16: #shrubbery 0 - 2.5
#                 lidar[x,i,j,] = 1
#             if point >= 16 and point < 32: #young 2.5 - 5
#                 lidar[x,i,j,] = 2
#             if point >= 32 and point < 77: #adult 5 - 12
#                 lidar[x,i,j,] = 3
#             if point >= 77 and point < 128: #very tall 12 - 20
#                 lidar[x,i,j,] = 4
#             if point >= 128: #extreme >20
#                 lidar[x,i,j,] = 5
# #height multiplier = 0.15

#'efficientnetb0' 0.0817 
# seresnet18 0.0779
#resnet18 0.0754
# seresnext50 0.0745
# senet154 0.0729
#resnet101 0.0700
#densenet121 0.0650
#inceptionresnetv2 0.0606
# vgg16 0.0569

#Linknet
# vgg16 0.1083
# efficientnetb0 0.0892
# densenet121 0.0769
# resnet101 0.0751
# inceptionresnetv2 0.0674

#FPN
# densenet121 0.1663
# inceptionresnetv2 0.0782
# efficientnetb0 0.0750
# resnet101 0.0701
# vgg16 0.0398 3e-5 bs8

#PSPNet
# efficientnetb0 0.0893
# vgg16 0.0843
# inceptionresnetv2 0.0775
# resnet101 0.0762 3e-5
# densenet121 0.0754

