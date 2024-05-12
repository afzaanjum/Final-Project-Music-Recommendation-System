## Group Members:
Ali Tanvir: 21I-1692
Afza Anjum: 21I-1724
Ruwaid Imran: 21I-1728

# Final Project: Music Recommendation System

## Phase 1: Extract, Transform, Load (ETL) Pipeline for Free Music Archive (FMA) Dataset

### Introduction:
The process of creating an ETL pipeline for the Free Music Archive (FMA) dataset, focusing on extracting audio features, transforming them into numerical formats,
and loading them into a MongoDB database. The dataset consists of 106,574 tracks, each lasting 30 seconds, and spans 161 genres. The goal is to prepare the dataset 
for music recommendation applications by extracting important features such as Mel-Frequency Cepstral Coefficients (MFCC), spectral centroid, and zero-crossing rate.


### Data Extraction and Preparation:
The initial step of the project involved extracting audio files from the provided dataset, which was compressed in a ZIP format. 
The dataset, known as fma_large.zip, contains a total of 106,574 tracks, each lasting 30 seconds, and covers 161 unevenly distributed genres. 
The dataset also includes additional metadata in the fma_metadata.zip file, providing details such as track titles, artist names, genres, tags, and 
play counts for all 106,574 tracks. Upon extraction, the dataset was stored in a local directory for further processing.

### Feature Extraction:
The next step focused on extracting key features from the audio files using the librosa library in Python. 
Three main features were extracted for each audio file:

### Mel-Frequency Cepstral Coefficients (MFCC):
MFCCs are widely used in audio signal processing and provide a compact representation of the audio spectrum.
Spectral Centroid: This feature represents the center of mass of the audio spectrum and provides information about the brightness of the sound.
Zero-Crossing Rate: This feature measures the rate at which the audio signal changes from positive to negative or vice versa and is often used to estimate the pitch of the audio.

### Data Transformation and Normalization:
After extracting the features, normalization was applied to ensure that all features were on a consistent scale. StandardScaler from the sklearn.preprocessing module
was used for this purpose. Normalization helps in improving the performance of machine learning models by ensuring that all features contribute equally to the analysis.

### Dimensionality Reduction:
To reduce the dimensionality of the MFCC features and improve computational efficiency, Principal Component Analysis (PCA) was applied. PCA helps in capturing the most 
important aspects of the data while reducing its complexity. In this project, PCA was used to reduce the dimensionality of the MFCC features to 10 components.

### Data Loading into MongoDB:
Finally, the extracted and transformed features were loaded into a MongoDB database using the pymongo library. Each document in the MongoDB collection contains the filename of the audio file,
the PCA-transformed MFCC features, the spectral centroid, and the zero-crossing rate.


## Phase 2: Music Recommendation Model

In this phase, the focus is on training a music recommendation model using Apache Spark. The goal is to utilize the data stored in MongoDB to build a model that can provide 
accurate music recommendations to users. The implementation involves the following steps:

### Connect to MongoDB:
Initially, a connection is established to the MongoDB database where the audio features are stored. 
This allows for easy retrieval of the data for model training.

### Data Preparation:
The audio features retrieved from MongoDB are prepared for training the model. This involves assembling the features into a vector 
format that can be used by the machine learning algorithm.

### Model Selection:
For the music recommendation model, a Random Forest classifier is chosen. This classifier is well-suited for this task as it can handle
multiple features and is known for its accuracy in classification tasks.

### Model Training:
The model is trained using the training data, which is a subset of the audio features retrieved from MongoDB.
The training process involves fitting the Random Forest classifier to the training data.

### Model Evaluation:
Once the model is trained, it is evaluated using the testing data, which is another subset of the audio features.
The evaluation is done using the MulticlassClassificationEvaluator, which calculates metrics such as accuracy, precision, recall, and F1-score.

 ### Hyperparameter Tuning:
Hyperparameters of the Random Forest classifier are tuned to improve the model's performance. This step is crucial 
as the selection of hyperparameters can greatly affect the model's accuracy.

Overall, this phase focuses on building and evaluating a music recommendation model using Apache Spark and MongoDB. 
The goal is to develop a model that can provide accurate recommendations to users based on the audio features extracted from the FMA dataset.
