#!/usr/bin/env python

# METADATA

__author__ = "Adrian Rodriguez-Bazaga, Josep M. Porta"
__credits__ = ["Adrian Rodriguez-Bazaga", "Josep M. Porta"]
__version__ = "1.0.0"
__email__ = "adrianrodriguezbazaga@gmail.com"

# PACKAGES

# Basic system routines packages
import os, sys

# Others
import numpy as np
import javabridge
import bioformats
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import scipy.misc

def save_tif(image, name, new_dir):
    new_name = os.path.join(new_dir, name)
    scipy.misc.imsave(new_name, image)

def runLIFtransformation(fileName, show=False, w=1024, h=1024):
    fileNameDir = fileName.split("/")
    fileNameDir.pop()
    fileNameDir = "/".join(fileNameDir)
    
    fileNameVar = fileName.split("/")[-1]
    path = fileName.split("/")
    path.pop()
    path = "/".join(path)
    path = fileName
    imagedataarray = np.zeros((h, w, 3), dtype=np.float32)

    javabridge.start_vm(class_path=bioformats.JARS)
    classpath = javabridge.JClassWrapper('java.lang.System').getProperty('java.class.path')

    with bioformats.ImageReader(path) as f:
        zValues = [0,1,2,3,4,5,6,7,8,9]

        for zVal in zValues:
            data = f.read(z = zVal, rescale=True)
            channel1data = data[:,:,0]
            channel2data = data[:,:,1]
            imagedataarray[:,:,2] += channel1data
            imagedataarray[:,:,1] += channel2data

        imagedataarray[:,:,2] *= 5
        imagedataarray[:,:,1] *= 5

    javabridge.kill_vm()

    # Show
    if show:
        plt.imshow(imagedataarray)
        plt.show()

    # Save
    fileNameGoodExtension = fileNameVar.replace(".lif", ".tif")
    print(fileNameDir)
    save_tif(imagedataarray, fileNameGoodExtension, fileNameDir)


# Just in case you want to use this as standalone script
'''
if __name__ == '__main__':
    # Ask for file
    app = QApplication([])
    fileName = QFileDialog.getOpenFileName(caption='Select a LIF file to be converted to TIF',filter='*.lif')[0]
    
    # Run script
    runLIFtransformation(fileName = fileName, show = True)
'''
