# Collagen VI-related Muscular Dystrophies Diagnosis

Convolutional Neural Network model and diagnosis system implementation of the manuscript "A Convolutional Neural Network for the Automatic Diagnosis of Collagen VI related Muscular Dystrophies", Bazaga et al. (Submitted to Expert Systems With Applications).

# Abstract

The development of machine learning systems for the diagnosis of rare diseases is challenging mainly due the lack of data to study them. Despite this challenge, this paper proposes a system for the Computer Aided Diagnosis (CAD) of low-prevalence, congenital muscular dystrophies from confocal microscopy images. The proposed CAD system relies on a Convolutional Neural Network (CNN) which performs an independent classification for non-overlapping patches tiling the input image, and generates an overall decision summarizing the individual decisions for the patches on the query image. This decision scheme points to the possibly problematic areas in the input images and provides a global quantitative evaluation of the state of the patients, which is fundamental for diagnosis and to monitor the efficiency of therapies.

# Overview of the system

The system is divided in four modules. The first module receives a full image and splits it into non-overlapping patches of 64x64 pixels. The second module is formed by the CNN classification model, that receives the patches and outputs an independent prediction for each one of them. The third module receives the local decisions for each patch and takes a global decision using majority voting. The last module visualizes the input image, the decision for each patch represented in a color code, and the overall decision of the system (see Fig. 4). Cyan is used to frame patches with more than 90% probability of belonging to the control class, steel blue is used for patches with
probability between 70% and 90%, yellow for patches with probability between 50% and 70%, orange for patches with probability between 30% and 50%, and finally red for patches with less than 30% of probability of belonging to the control class. This color code offers the possibility of easily spotting suspicious areas in the image. The system also provides the overall decision on the image and a global score computed as the percentage of patches classified as control in the image.

![Overview](https://github.com/AdrianBZG/Muscular-Dystrophy-Diagnosis/blob/master/Media/ReadmeFig2.png)

# Example images

![Example Images](https://github.com/AdrianBZG/Muscular-Dystrophy-Diagnosis/blob/master/Media/ReadmeFig1.png)


# Preprint

Although the manuscript has been submitted for review to Expert Systems With Applications, a preprint has been available available at [https://arxiv.org/abs/1901.11074](https://arxiv.org/abs/1901.11074).
