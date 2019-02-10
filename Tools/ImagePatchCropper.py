import numpy
import os, sys
import tifffile as tiff

inputDirectory = sys.argv[1]
outputDirectory = sys.argv[2]

import image_slicer

for file in os.listdir(inputDirectory):
    if file.endswith(".tif"):
        filePath = os.path.join(inputDirectory, file)
        patches = image_slicer.slice(filePath, 256, save=False)
        image_slicer.save_tiles(patches, directory=outputDirectory, prefix=file.split(".tif")[0]+'_patch', format='tiff')
