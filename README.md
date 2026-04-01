# CEG3004 Audio Classification Project  Pr_33

## Overview
This project implements an audio classification pipeline for the ESC-50 environmental sound dataset (50 classes, 2000 clips). The pipeline extracts hand-crafted DSP features from audio waveforms and classifies them using a Support Vector Machine (SVM) with an RBF kernel. The system is evaluated on clean, noisy, and band-limited versions of the submission set to assess robustness under realistic signal distortions.


## Objective
The main objectives are:  
• Train on labelled environmental sound data  
• Extract meaningful DSP features  
• Use a machine learning classifier to classify into the 50 sound classes  
• Demonstrates robustness to noise and bandwidth distortions  

---

### Dataset  
- **Total clips:** 2,000
- **Number of classes:** 50
- **Clips per class:** 40
- **Clip duration:** 5 seconds
- **Audio format:** mono waveform

The dataset is divided into:

- **Labelled training set** for model development
- **Unlabelled submission set** for final evaluation

Each submission clip is provided in three forms:

- **Clean** – original signal
- **Noisy** – additive noise applied
- **Band-limited** – frequency content restricted

These variations are intended to evaluate how robust the DSP pipeline and classifier are under realistic distortions.

---

### Pipeline of the project  
• Setup  
• Dataset loading  
• Preprocessing  
• Feature extraction  
• Train/validation split  
• Model training  
• Evaluation  
• Submission prediction  


## Methodology  
The overall workflow of the project is:
1. Load the labelled training audio
2. Apply preprocessing to standardize the waveforms
3. Extract DSP-based features from each clip
4. Train and validate multiple machine learning models
5. Compare validation performance using Macro-F1
6. Select the best-performing model
7. Generate predictions for the submission set


## Audio Preprocessing

Before feature extraction, each audio clip was preprocessed to make the signals more consistent and suitable for analysis.

The preprocessing steps were:
- Replace invalid numerical values using `np.nan_to_num`
- Remove DC offset by centering the waveform around zero
- Apply peak normalization to reduce amplitude variation
- Apply a pre-emphasis filter to strengthen higher-frequency content
- Pad or truncate each clip to a fixed duration of **5 seconds**

These steps helped reduce unnecessary variation across clips and improved the stability

## Feature Extraction
A set of DSP-based features was extracted from each audio clip to represent its acoustic characteristics in a compact numerical form.

The following audio features were extracted to capture both the spectral and temporal characteristics of the environmental sound signals:

| Feature | What it captures | Why it is useful |
|---|---|---|
| **MFCCs** | Overall spectral shape and timbre based on human auditory perception | Helps distinguish sound classes using tonal and frequency patterns |
| **Delta MFCCs** | First-order temporal change of MFCCs | Adds information about how the sound evolves over time |
| **Delta-delta MFCCs** | Second-order temporal change of MFCCs | Captures more detailed dynamic behaviour |
| **Log-mel spectrogram statistics** | Distribution of energy across mel-scaled frequency bands | Provides a compact time-frequency representation |
| **Spectral centroid** | Centre of mass of the spectrum | Indicates whether a sound is brighter or darker |
| **Spectral bandwidth** | Spread of frequencies around the centroid | Describes whether the sound is narrow-band or wide-band |
| **Spectral rolloff** | Frequency below which most spectral energy is concentrated | Helps identify high-frequency content |
| **Zero-crossing rate** | Number of zero crossings in the waveform | Useful for noisy, percussive, or sharp sounds |
| **RMS energy** | Overall signal energy | Helps represent loudness and intensity |

For each feature type, summary statistics such as **mean**, **standard deviation**, and **median** were computed across time frames to produce a fixed-length feature vector for classification.

## Data Augmentation
To improve robustness, lightweight audio augmentation was applied during training.

The augmentation pipeline included:

- Mild random gain scaling
- Optional white Gaussian noise
- Optional Butterworth-based filtering
- Optional bandwidth limitation via resampling

Only mild augmentation was used so that the model could learn to handle distorted audio without drifting too far away from the clean training distribution. The augmented clips were used to improve robustness, while the original clips remained part of the training set.

---

## Models Evaluated

Several machine learning models were evaluated on the extracted DSP feature vectors:

- **Logistic Regression** was used as the baseline classifier because it is simple, fast, and provides a useful reference point.
- **Random Forest** was tested as a tree-based ensemble model that can handle non-linear feature relationships.
- **SVM** was included because it often performs well on structured, hand-crafted audio features and can model non-linear decision boundaries.
- **KNN** was included as a distance-based classifier to compare how well samples cluster in the extracted feature space.
- **Extra Trees** was tested as another ensemble tree-based model that introduces additional randomness and can improve generalization.

To ensure a fair comparison, all candidate models were evaluated using the same extracted feature set, the same stratified train-validation split, and the same evaluation metric. This avoided inconsistencies that can arise when models are tested separately under different notebook states or random conditions.

The final model was selected based on validation performance, with Macro-F1 used as the main comparison metric.

| Model | Macro-F1 |
|------|----------|
| Logistic Regression | 0.7138 |
| Random Forest | 0.6525 |
| SVM | 0.7152 |
| KNN | 0.4669 |
| Extra Trees | 0.6553 |

Among the evaluated models, **SVM achieved the highest Macro-F1 score of 0.7152**, outperforming the other classical machine learning models. This indicates that SVM was the most effective at separating the different sound classes in the extracted DSP feature space.
### Final Model Selection
Based on the validation results, **SVM was selected as the final model** for submission.

---

## Final Validation Summary
The final validation output achieved:

- **Accuracy:** 0.72
- **Macro-F1:** 0.7152
- **Weighted F1:** 0.72

The class-wise classification report showed that many classes performed strongly, while more ambiguous environmental sounds such as `washing_machine`, `wind`, and `vacuum_cleaner` remained more challenging. Overall, the results suggest that the DSP feature set and SVM classifier were able to capture useful distinctions across a wide range of environmental sound classes.

---

## Why SVM Was Chosen
SVM was chosen as the final model because it performed best on the extracted hand-crafted DSP features. Compared with the other tested models, it showed stronger class separation and better overall Macro-F1 performance.

This is consistent with the fact that SVM often works well in high-dimensional feature spaces built from MFCCs, spectral statistics, and other structured audio descriptors.

---

## How to Run
1. Open the notebook in Google Colab
2. Install the required dependencies
3. Prepare the dataset in the expected folder structure
4. Run all cells from top to bottom
5. The notebook will generate the trained model file and prediction CSV

## Repository Structure
```text
.
├── CEG3004_Project_Colab.ipynb   # Main notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── results/                      # Generated outputs and plots
