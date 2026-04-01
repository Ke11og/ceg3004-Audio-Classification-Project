# CEG3004 Audio Classification Project  Pr_33

This project implements an audio classification pipeline for the ESC-50 environmental sound dataset (50 classes, 2000 clips). The pipeline extracts hand-crafted DSP features from audio waveforms and classifies them using a Support Vector Machine (SVM) with an RBF kernel. The system is evaluated on clean, noisy, and band-limited versions of the submission set to assess robustness under realistic signal distortions.

### Objective
The main objectives are:  
• Train on labelled environmental sound data  
• Extract meaningful DSP features  
• Use a machine learning classifier to classify into the 50 sound classes  
• Demonstrates robustness to noise and bandwidth distortions  

### Dataset  
The project uses an environmental sound classification dataset derived from the ESC-50 collection.   
The dataset contains 2,000 audio clips distributed across 50 sound classes, with 40 clips per class. Each clip is 5 seconds long and recorded as a single-channel (mono) waveform.
The dataset is split into a labelled training set and an unlabeled submission set for evaluation.   
The submission set contains three versions of each clip:  
• Clean (original signal)  
• Noisy (additive noise applied)  
• Band-limited (frequency content restricted)  
• All three versions correspond to the same underlying sound event but are designed to test the robustness of your DSP feature extraction and model design under realistic distortions.

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

The following audio features were extracted to capture both the spectral and temporal characteristics of the environmental sound signals:

| Feature | What it captures | Why it is useful |
|---|---|---|
| **MFCCs** | The overall spectral shape and timbre of the audio, based on human hearing characteristics | Helps distinguish different sound types by capturing tonal and frequency patterns |
| **Delta MFCCs** | The rate of change of MFCCs over time | Adds temporal information and shows how the sound evolves frame by frame |
| **Delta-delta MFCCs** | The acceleration of change of MFCCs over time | Captures more detailed dynamic behaviour of the sound |
| **Log-mel spectrogram statistics** | Summary statistics of energy distribution across mel-scaled frequency bands over time | Provides a compact representation of time-frequency energy patterns |
| **Spectral centroid** | The center of mass of the frequency spectrum | Indicates whether a sound is brighter or darker |
| **Spectral bandwidth** | The spread of frequencies around the spectral centroid | Describes whether the sound is narrow-band or wide-band |
| **Spectral rolloff** | The frequency below which most of the spectral energy is concentrated | Helps identify how much high-frequency content is present |
| **Zero-crossing rate** | How often the waveform crosses the zero-amplitude axis | Useful for distinguishing noisy or percussive sounds from smoother tonal sounds |
| **RMS energy** | The overall signal energy or loudness | Helps measure intensity and separate quiet from loud sounds |

For each feature type, summary statistics such as mean, standard deviation, and median were computed across frames to form a fixed-length feature vector for classification.  

## Models Evaluated

Several machine learning models were evaluated on the extracted DSP feature vectors:

- **Logistic Regression** was used as the baseline classifier because it is simple, fast, and provides a useful reference point.
- **Random Forest** was tested as a tree-based ensemble model that can handle non-linear feature relationships.
- **SVM** was included because it often performs well on structured, hand-crafted audio features and can model non-linear decision boundaries.
- **Gradient Boosting** was tested as another ensemble-based method that iteratively improves classification performance.
- **KNN** was included as a distance-based classifier to compare how well samples cluster in the extracted feature space.
- **Extra Trees** was tested as another ensemble tree-based model that introduces additional randomness and can improve generalization.

To ensure a fair comparison, all candidate models were evaluated using the same extracted feature set, the same stratified train-validation split, and the same evaluation metric. This avoided inconsistencies that can arise when models are tested separately under different notebook states or random conditions.

The final model was selected based on validation performance, with Macro-F1 used as the main comparison metric.

| Model | Macro-F1 |
|------|----------|
| Logistic Regression | 0.4972 |
| Random Forest | 0.5456 |
| SVM | 0.5902 |
| KNN | 0.3970 |
| Extra Trees | 0.5645 |

Among the evaluated models, SVM achieved the highest Macro-F1 score of 0.5902, followed closely by Extra Trees at 0.5645. This suggests that SVM performed best at capturing the non-linear relationships in the extracted DSP feature space, likely due to its effectiveness in high-dimensional settings. The baseline was decent but clearly weaker. KNN did not work well for this feature space, hence **SVM is chosen as the final model**.

## Augmentation 
The best practical setup is:

Ror each training clip:  
• Always extract features from the original waveform  
• Optionally extract features from one augmented version  
• Do not augment validation clips  
• Do not stack too many augmentations on the same clip at first  

Use:  
• Mild Gaussian noise  
• Mild gain scaling  
• Random lowpass / bandpass effect

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
