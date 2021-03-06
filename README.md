# UWE Digital Systems Project
This repository is the result of my final year project for my dissertation at the university of the west of england. 

## Project Aim

This project is aiming at helping people & researchers understand audio signals and its classification by visualising the signal as well as evaluating state of the art audio feature extraction and classification methods.

## Project Objectives

* Research into existing methods of feature extraction for audio.
* Research into neural networks for audio classification
* Identify a good performing neural network architecture for audio classification
* Design, build and, implement a cross platform interface program to visualize audio signals with at least 2 latest feature extraction methods (MFCC, Spectrograms, Mel Spectrograms).
* Train Existing Neural Network Architectures on the Kaggle2018 Dataset.
* Compare and analyse different neural network architectures (with the same dataset) in conjunction with different feature extraction methods.
* Identify the best hyper parameters for different neural network architectures that yield best accuracy for audio classification.
* Identify the advantages and limitations of each used feature extraction method.

## CNN Architectures
This research project compares each of the following convolution neural network architectures:
<br>
<br>
<p align="center">
    <img src="https://raw.githubusercontent.com/Auzlex/uwe-dsp/main/documents/nn_models.png" width=70%>
</p>

## Directory
Brief directory overview
```
project
│   README.md
│   requirements.txt    
└───app:
│   │   execute.pyw -> runs the interface program
│   │   config.py -> configures the audio related settings 
│   └─── tf_models:
│       │     this folder contains old redacted trained models that are not good at all
│       │     You may find the fully trained example models on Google Drive
└───documents:
│   │   
│   └─── designs: (designs related to the research project)
│   │   
│   └─── research: (related pdfs, some may be missing)
└───notebooks:
│   │   Data_Normalizer_Cacher.ipynb (normalizes to npy and inspects dataset)
│   │   Data_Feature_Extractor.ipynb (handles feature extraction and caches them as npy files)
│   │   helper.py (contains helper functions used by notebooks)
│   └─── MEL: (mel spectrogram based CNN training notebooks)
│   │   
│   └─── MFCC: (MFCC based CNN training notebooks)
└───redacted_notebooks: 

    contains all the redacted notebooks that where essential to the learning of tensorflow 2.0 but may be incomplete or in disarray as they where retired.
```

H5 Model Attributes to work with interface program
```
└───h5:
│   │   
│   └─── metadata: (array of strings that are properly in encoding order to match softmax layer)
│   │   
│   └─── dfe: the desired feature extraction method the model wants
```
The attributes are attached with the h5 model refer to the helper.py load & save ext functions, they tell the interface program what the model expects, granted not perfect but sufficient to meet the deadline of this research project.

## Interface program
This is an old screenshot of the interface program running a simple CNN model classifying audio sounds. In this example, the interface program is showing a linear-scale spectrogram, a mel scaled variant and a MFCC. The bar chart shows the multiple labels that can be predicted by the model as the metadata is embedded within it.
<p align="center">
    <img src="https://raw.githubusercontent.com/Auzlex/uwe-dsp/main/documents/python3.8_2022-03-31_14-15-48.png" width=70%>
</p>

The interface program will function and switch input devices on both windows and the Linux distribution Fedora 35(only tested linux distribution)

## Trained Models

Trained models can be found on Google Drive here:

* I have yet to upload them to google drive as I am in the middle of reorganizing my personal computers storage
