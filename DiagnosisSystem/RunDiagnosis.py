#!/usr/bin/env python

# METADATA

__author__ = "Adrian Rodriguez-Bazaga, Josep M. Porta"
__credits__ = ["Adrian Rodriguez-Bazaga", "Josep M. Porta"]
__version__ = "1.0.0"
__email__ = "adrianrodriguezbazaga@gmail.com"

# PACKAGES

# Basic system routines packages
import os, sys

# Keras imports
from keras import *
from keras.preprocessing import image
from keras.layers import *
from keras.models import *

# Matplotlib and colors
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patchesMatplotlib

# Others
import numpy as np
import ntpath
import tifffile as tiff
import image_slicer
import shutil
from PyQt5.QtWidgets import QApplication, QFileDialog

# Add the Utils folder path to the sys.path list
sys.path.append('../Tools/')

# Import LIF to TIF transformation script
from LIFTransformationUtils import runLIFtransformation

# Import model custom metrics
from CustomModelMetrics import m_recall
from CustomModelMetrics import m_f1
from CustomModelMetrics import m_precision

# Import helper functions
from HelperFunctions import load_image
from HelperFunctions import flatten

# SCRIPT
if __name__ == '__main__':
    app = QApplication([])

    # Ask for weights
    weightsFile = QFileDialog.getOpenFileName(caption='Select a h5 weights file',filter='*.h5')[0]
    print(weightsFile)

    # Ask for file
    fileName = QFileDialog.getOpenFileName(caption='Select a LIF file',filter='*.lif')[0]
    print(fileName)

    runLIFtransformation(fileName = fileName, show = False)

    fileNameTif = fileName
    fileNameTif = fileNameTif.replace(".lif", ".tif")

    inputImage = fileNameTif

    # Chop the image in patches and store it in a temp directory
    tempDir = os.path.dirname(os.path.abspath(__file__)) + "/TempDir/"

    if os.path.isdir(tempDir):
        shutil.rmtree(tempDir)

    os.mkdir(tempDir)

    patches = image_slicer.slice(inputImage, 256, save=False)
    image_slicer.save_tiles(patches, directory=tempDir, prefix=inputImage.split(".tif")[0]+'_patch', format='tiff')

    for root, subFolders, files in os.walk(os.path.dirname(os.path.abspath(inputImage))):
        for file in files:
            if file.endswith(".tiff"):
                if not os.path.isfile(tempDir+file):
                    shutil.move(os.path.join(root, file), tempDir)
                    os.rename(tempDir+file, tempDir+file.replace("tiff","tif"))

    # Load the CNN
    print("* Loading model...")
    model = load_model(weightsFile, custom_objects={'m_recall': m_recall, 'm_precision': m_precision, 'm_f1': m_f1})
    print("* Model loaded")

    print("* Starting diagnosis...")

    # Predicting the images in the TempDir
    imagesToPredict = []
    imagesToPredictNames = []

    # Add each patch to be predicted to the list
    for file in os.listdir(tempDir):
        if file.endswith(".tif"):
            imagesToPredictNames.append(file)

    # Sort them, to have in order for further usage in the plot
    imagesToPredictNames.sort()

    # Load each image to be predicted in matrix form to the array
    for file in imagesToPredictNames:
        imagesToPredict.append(load_image(os.path.join(tempDir, file)))

    # Create a vertical stack from the images list
    images = np.vstack(imagesToPredict)

    # Predict classes of the images and retain the probabilities
    classes = model.predict_classes(images)
    classesProb = model.predict(images)

    predictions = flatten(classes)
    predictionsProbs = classesProb

    # Prediction text (for the top of the plot) for this full image
    predictionText = "Goodness: " + str(round((len(predictions) - np.count_nonzero(predictions))/len(predictions)*100,1)) + "%"

    if np.count_nonzero(predictions) > (len(predictions) - np.count_nonzero(predictions)):
        predictionText += " (Class: Patient)"
    elif (len(predictions) - np.count_nonzero(predictions)) > np.count_nonzero(predictions):
        predictionText += " (Class: Control)"
    else:
        predictionText += " (Unsure)"

    # Remove the temp directory and its content recursively
    shutil.rmtree(tempDir)

    # Show image and colored patches according to probability
    f, ax = plt.subplots(1, 1)
    ax.set_title(ntpath.basename(inputImage) + " - Diagnosis: " + predictionText)
    ax.imshow(load_image(inputImage, (1024,1024))[0])

    # Add a rectangle for each patch in the image with its color according to the probability of belonging to the control class
    classIndex = 0 # Default
    classColor = 'black' # Default

    for x in (0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960):
        for y in (0,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960):
            if(predictionsProbs[classIndex][0] >= 0.9): classColor = "cyan"
            elif(predictionsProbs[classIndex][0] >= 0.7 and predictionsProbs[classIndex][0] < 0.9): classColor = "steelblue"
            elif(predictionsProbs[classIndex][0] >= 0.5 and predictionsProbs[classIndex][0] < 0.7): classColor = "yellow"
            elif(predictionsProbs[classIndex][0] >= 0.3 and predictionsProbs[classIndex][0] < 0.5): classColor = "orange"
            else: classColor = "red"

            rect = patchesMatplotlib.Rectangle((x,y),62,62,linewidth=1,edgecolor=classColor,facecolor='none')
            ax.add_patch(rect)

            classIndex = classIndex+1

    plt.axis('off')

    print("* Diagnosis finished")
    plt.show()