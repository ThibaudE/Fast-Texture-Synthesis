
# Texture synthesis with patch optimal transport

B. Galerne, A. Leclaire, J. Rabin (c) 2019

 This repository contains Python code related to the paper

[**A Texture Synthesis Model Based on Semi-discrete Optimal Transport in Patch Space**](https://doi.org/10.1137/18M1175781)  
Bruno Galerne, Arthur Leclaire, Julien Rabin,  
SIAM Journal on Imaging Sciences, vol. 11, no. 4, pp. 2456-2493, 2018.  
[Project page](https://www.math.u-bordeaux.fr/~aleclaire/texto/).
[Hal preprint](https://hal.archives-ouvertes.fr/hal-01726443)

## List of files

The main files are:
* texto.py : contains functions to estimate/synthesize the texture model
* texto_test.py : script file to estimate/synthesize one texture with visualization
* texto_loop.py : script file to process many textures in batch

Miscellaneous
* gaussian_texture.py : functions to estimate/synthesize a Gaussian texture model
* patch.py : define the patch classes (patch extractor, and pseudo-inverse)
* semi_discrete_ot.py : functions to estimate/sample semi-discrete optimal transport maps

## Example Usage

```
model = texto.model(im0, 3, 4, 4)   
synth = model.synthesize(512, 768)
```

In this example, the first line estimates the texture model with 4 scales, patches 3x3, 4 Gaussian components at each scale. The second line performs synthesis of size 768x512.

You can further test these functions by executing the script texto_test.py .

## Pre-estimated textures

For certain textures, pre-estimated models are available in the folder models/ .

## Other models
You may find other tested models in the different subfolders, mainly :
* multires : From "Multi-resolution Texture Synthesis with CNN" paper, see readme in the folder.
* quilting : From "Image Quilting for Texture Synthesis", see readme in the folder.

Those are the two additions to the presentation, especially the first one from which we could perform Gatys image pyramid optimization.