# Collagen VI-related Muscular Dystrophies Diagnosis

Convolutional Neural Network model and diagnosis system implementation of the paper "A Convolutional Neural Network for the automatic diagnosis of collagen VI-related muscular dystrophies", Bazaga et al. (Accepted in Applied Soft Computing).

# How to cite this software

[![DOI](https://zenodo.org/badge/DOI/10.1016/j.asoc.2019.105772.svg)](https://doi.org/10.1016/j.asoc.2019.105772)

# Abstract

The development of machine learning systems for the diagnosis of rare diseases is challenging, mainly due to the lack of data to study them. This paper surmounts this obstacle and presents the first Computer-Aided Diagnosis (CAD) system for low-prevalence collagen VI-related congenital muscular dystrophies. The proposed CAD system works on images of fibroblast cultures obtained with a confocal microscope and relies on a Convolutional Neural Network (CNN) to classify patches of such images in two classes: samples from healthy persons and samples from persons affected by a collagen VI-related muscular distrophy. This fine-grained classification is then used to generate an overall diagnosis on the query image using a majority voting scheme. The proposed system is advantageous, as it overcomes the lack of training data, points to the possibly problematic areas in the query images, and provides a global quantitative evaluation of the condition of the patients, which is fundamental to monitor the effectiveness of potential therapies. The system achieves a high classification performance, with 95% of accuracy and 92% of precision on randomly selected independent test images, outperforming alternative approaches by a significant margin.

# Overview of the system

The system is divided in four modules. The first module receives a full image and splits it into non-overlapping patches of 64x64 pixels. The second module is formed by the CNN classification model, that receives the patches and outputs an independent prediction for each one of them. The third module receives the local decisions for each patch and takes a global decision using majority voting. The last module visualizes the input image, the decision for each patch represented in a color code, and the overall decision of the system (see Fig. 4). Cyan is used to frame patches with more than 90% probability of belonging to the control class, steel blue is used for patches with
probability between 70% and 90%, yellow for patches with probability between 50% and 70%, orange for patches with probability between 30% and 50%, and finally red for patches with less than 30% of probability of belonging to the control class. This color code offers the possibility of easily spotting suspicious areas in the image. The system also provides the overall decision on the image and a global score computed as the percentage of patches classified as control in the image.

![Overview](https://github.com/AdrianBZG/Muscular-Dystrophy-Diagnosis/blob/master/Media/ReadmeFig2.png)

# Example images

![Example Images](https://github.com/AdrianBZG/Muscular-Dystrophy-Diagnosis/blob/master/Media/ReadmeFig1.png)


# Citation

You can cite our [https://doi.org/10.1016/j.asoc.2019.105772](paper) using the following BibTeX item:

`@article{BAZAGA2019105772,
title = "A Convolutional Neural Network for the automatic diagnosis of collagen VI-related muscular dystrophies",
journal = "Applied Soft Computing",
volume = "85",
pages = "105772",
year = "2019",
issn = "1568-4946",
doi = "https://doi.org/10.1016/j.asoc.2019.105772",
url = "http://www.sciencedirect.com/science/article/pii/S1568494619305538",
author = "Adrián Bazaga and Mònica Roldán and Carmen Badosa and Cecilia Jiménez-Mallebrera and Josep M. Porta",
keywords = "Convolutional neural networks, Deep learning, Classification, Computer aided diagnosis, Confocal microscopy images",
abstract = "The development of machine learning systems for the diagnosis of rare diseases is challenging, mainly due to the lack of data to study them. This paper surmounts this obstacle and presents the first Computer-Aided Diagnosis (CAD) system for low-prevalence collagen VI-related congenital muscular dystrophies. The proposed CAD system works on images of fibroblast cultures obtained with a confocal microscope and relies on a Convolutional Neural Network (CNN) to classify patches of such images in two classes: samples from healthy persons and samples from persons affected by a collagen VI-related muscular distrophy. This fine-grained classification is then used to generate an overall diagnosis on the query image using a majority voting scheme. The proposed system is advantageous, as it overcomes the lack of training data, points to the possibly problematic areas in the query images, and provides a global quantitative evaluation of the condition of the patients, which is fundamental to monitor the effectiveness of potential therapies. The system achieves a high classification performance, with 95% of accuracy and 92% of precision on randomly selected independent test images, outperforming alternative approaches by a significant margin."
}`
