# Recognition of Musical Instruments based on Mel-Frequency Cepstral Analysis
This project presents a machine learning pipeline for musical instrument recognition based on Mel-Frequency Cepstral Coefficients (MFCC) and additional spectral features. 
Dimensionality reduction techniques (LDA and LFDA) were applied to improve class separability and classification performance. 
The final classification was performed using a K-Nearest Neighbors (KNN) classifier.


## Problem Description
The goal of this project is to automatically classify musical instruments based on their audio recordings. 
The task is formulated as a multi-class classification problem.


## Feature Extraction
For each audio sample, the following features were extracted:

- 13, 16, 20 MFCC coefficients
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Roll-off

For each feature, mean and standard deviation were computed across time frames, resulting in a fixed-length feature vector (34–48 dimensions depending on the number of MFCC coefficients).


## Preprocessing
All features were standardized using StandardScaler to ensure zero mean and unit variance before applying dimensionality reduction and classification.


## Dimensionality Reduction
Two supervised dimensionality reduction methods were applied:

- Linear Discriminant Analysis (LDA)
- Local Fisher Discriminant Analysis (LFDA)

LDA reduces the dimensionality to at most (number of classes − 1), while LFDA preserves local structure and allows more flexible projections.


## Classification
Classification was performed using the K-Nearest Neighbors (KNN) algorithm with Euclidean distance. 
The influence of the number of neighbors (k) on classification accuracy was analyzed.


## Distance Metrics
The K-Nearest Neighbors classifier was evaluated using three different distance metrics in order to analyze their impact on classification performance:

### Euclidean distance (L2 norm) – the standard metric used in KNN, based on the straight-line distance between feature vectors.

### Manhattan distance (L1 norm) – computes the sum of absolute differences between feature components.

### Mahalanobis distance – accounts for feature correlations and incorporates the covariance structure of the data, which is particularly relevant for MFCC-based feature representations.

For each distance metric, a separate implementation file is provided in the repository, allowing independent evaluation and comparison of classification performance.

This comparative analysis highlights how the geometry of the feature space influences multi-class audio classification.


## Results
The best performance was achieved using 16 MFCC coefficients and Euclidean distance, reaching up to 85% classification accuracy.
Confusion matrices and performance comparison plots are included in the repository.


## Technologies
- Python
- Librosa
- NumPy
- Scikit-learn
- Matplotlib




This project was developed as part of my Bachelor Thesis in Electrical Engineering (Signals and Systems).
