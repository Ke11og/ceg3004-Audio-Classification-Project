# CEG3004 Audio Classification Project
The aim of this project is to design a robust audio classification pipeline for Environmental Sound Classification that performs well under clean and distorted conditions.

## Objective
The main objectives are:  
• Train on labelled environmental sound data  
• Extract meaningful DSP features  
• Use a machine learning classifier to classify into the 50 sound classes  
• Demonstrates robustness to noise and bandwidth distortions  

## Dataset  
The project uses an environmental sound classification dataset derived from the ESC-50 collection.   
The dataset contains 2,000 audio clips distributed across 50 sound classes, with 40 clips per class. Each clip is 5 seconds long and recorded as a single-channel (mono) waveform.
The dataset is split into a labelled training set and an unlabeled submission set for evaluation.   
The submission set contains three versions of each clip:  
• Clean (original signal)  
• Noisy (additive noise applied)  
• Band-limited (frequency content restricted)  
• All three versions correspond to the same underlying sound event but are designed to test the robustness of your DSP feature extraction and model design under realistic distortions.
