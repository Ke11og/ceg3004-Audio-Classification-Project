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

## Pipeline of the project  
• Setup  
• Dataset loading  
• Preprocessing  
• Feature extraction  
• Train/validation split  
• Model training  
• Evaluation  
• Submission prediction  


## Methodology  
The overall workflow of the project was:
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
- Invalid numerical values were replaced using `np.nan_to_num`
- DC offset was removed by centering the waveform around zero
- Leading and trailing silence were trimmed
- Peak normalization was applied so that clips had more consistent amplitude
- A pre-emphasis filter was used to highlight higher-frequency details
- Each clip was padded or truncated to a fixed duration of 5 seconds

These steps helped reduce unnecessary variation across clips and improved the stability

## Feature Extraction
A set of DSP-based features was extracted from each audio clip to represent its acoustic characteristics in a compact numerical form.

The extracted features included:
- MFCCs
- Delta MFCCs
- Delta-delta MFCCs
- Log-mel spectrogram statistics
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero-crossing rate
- RMS energy

For each feature type, summary statistics such as mean, standard deviation, and median were   computed across frames to form a fixed-length feature vector for classification.  

## Models Evaluated

Several machine learning models were evaluated on the extracted DSP feature vectors:

- **Logistic Regression** was used as the baseline classifier because it is simple, fast, and provides a useful reference point.
- **Random Forest** was tested as a tree-based ensemble model that can handle non-linear feature relationships.
- **SVM** was included because it often performs well on structured, hand-crafted audio features and can model non-linear decision boundaries.
- **Gradient Boosting** was tested as another ensemble-based method that iteratively improves classification performance.
- **KNN** was included as a distance-based classifier to compare how well samples cluster in the extracted feature space.
- **Extra Trees** was tested as another ensemble tree-based model that introduces additional randomness and can improve generalization.
  
The final model was selected based on validation performance, with Macro-F1 used as the main comparison metric.

| Model | Macro-F1 |
|------|----------|
| Logistic Regression | 0.4068 |
| Random Forest | 0.5460 |
| SVM & Gradient Boost| 0.4768 |
| KNN | 0.30648421948421944 |
| Extra Trees | 0.5524 |

Among the evaluated models, Extra Trees achieved the highest Macro-F1 score of 0.5524, followed closely by Random Forest at 0.5460. This suggests that tree-based ensembles methods performed best and were better at capturing the non-linear relationships in the extracted DSP feature space. The baseline was decent but clearly weaker. KNN did not work well for this feature space, hence the **Extra Trees is chosen as the final model.**


## How to Run
1. Open the notebook in Google Colab
2. Install the required dependencies
3. Prepare the dataset in the expected folder structure
4. Run all cells from top to bottom
5. The notebook will generate the trained model file and prediction CSV

## Repository Structure
- `CEG3004_Project_Colab.ipynb` — main project notebook
- `README.md` — project overview and documentation
- `requirements.txt` — required Python packages
- `results/` — generated plots and outputs
- `models/` — saved trained models
