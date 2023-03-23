"""
 Texture synthesis with patch optimal transport

 Execute this script to process all textures placed in the folder "tex/"

 Copyright Arthur Leclaire (c), 2019.

"""

import numpy as np
import matplotlib.pyplot as plt
import texto
import pickle
from os import listdir, mkdir
from os.path import isfile, isdir, join

if not isdir('synth'):
    mkdir('synth')
if not isdir('models'):
    mkdir('models')

files = [f for f in listdir('tex') if isfile(join('tex', f))]
files.sort()

plt.ion()

for ind in range(93, len(files)):
    name, ext = files[ind].rsplit('.', 1)

    print('\n')
    print('-----------------------------------------')
    print('----- Processing texture no ', ind, ' ', name,' -------')
    print('-----------------------------------------')
    print('\n')

    # Load input image
    im0 = np.double(plt.imread('tex/'+name+'.'+ext))
    if im0.ndim < 3:
        im0 = im0[:, :, np.newaxis]
    m, n, nc = im0.shape

    # Parameters
    w = 3
    nscales = 4
    ngmm = 4
    paramstr = '_w'+str(w)+'_nscales'+str(nscales)+'_ngmm'+str(ngmm)
    doestimation = True
    
    if doestimation:
        # Model estimation
        model = texto.model(im0, w, nscales, ngmm)
        # Save model in a file
        f = open('models/'+name+paramstr+'.pckl', 'wb')
        pickle.dump(model, f)
        f.close()

    # Load texture model from pre-computed file
    f = open('models/'+name+paramstr+'.pckl', 'rb')
    model = pickle.load(f)
    f.close()
        
    # Synthesis
    M, N = 512, 768
    synth = model.synthesize(M, N)

    # Save Results
    if nc == 1:
        synth = synth[:, :, 0]
        im0 = im0[:, :, 0]
    plt.imsave('synth/'+name+'_synth'+paramstr+'.'+ext, synth, cmap='Greys')
    plt.imsave('synth/'+name+'.'+ext, im0, cmap='Greys')
