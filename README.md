# Gaussian Probability Density Function Naïve Bayes Net for Music Genre Classification:

## Overview
This project leverages the Gaussian Naïve Bayes Net to classify music genres. Utilizing a comprehensive dataset that spans multiple genres, this implementation aims to predict the genre of a given piece of music based on its acoustic features.

## Dataset Overview:
The dataset for this classification task is the "Music Genre Classification", which comprises tracks across 11 genres including Acoustic/Folk, Alt Music, Blues, Bollywood, Country, Hip-Hop, Indie Alt, Instrumental, Metal, Pop, and Rock. Each track in the dataset is described by both categorical (e.g., Artist Name, Track Name, Mode) and numerical features (e.g., Popularity, Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration, Time Signature), with the target feature being the music genre class.

The dataset is publicly available on Kaggle and can be accessed here: https://www.kaggle.com/datasets/purumalgi/music-genre-classification/code.

## Exploratory Data Analysis:
Our exploratory data analysis (EDA) revealed diverse distributions across the numerical features, with some displaying normal distribution tendencies and others being skewed or multimodal. The dataset also exhibited class imbalance, with genres like Rock being more prevalent than Country or Bollywood. To address missing data within these features, median imputation was applied.

The EDA helped inform our preprocessing steps, ensuring the data was appropriately conditioned for effective model training and evaluation.

## Data Preprocessing:
In the preprocessing stage, we addressed skewed features through logarithmic transformations, utilizing the np.log1p function. This transformation was crucial for normalizing the distribution of certain features, thereby enhancing the model's predictive performance.

## Naïve Bayes Classifier Implementation:
The Gaussian Naïve Bayes classifier was custom-built and evaluated for its ability to classify music genres accurately. We divided the dataset into training and testing subsets, computed prior probabilities, and used the Gaussian PDF to estimate likelihoods. The model's predictive capability was tested across various metrics including accuracy, precision, recall, F1-score, and through a confusion matrix.

Despite certain limitations, such as the initial assumption of feature independence and distribution normality, the custom Naïve Bayes classifier demonstrated promising results, indicating its potential utility in music genre classification tasks. 

## Results:
Although our accuracy ended up being low (43.9%), this is on par with advanced tools that also implement Naïve Bayes Nets, such as Scikit-Learn which caps at 44.5% in our trials. Which shows that our implementation was good, just sadly not the right approach to the problem at hand.

## Prerequisites:
Summed up in the requirements.txt

## Installation:
pip3 install -r requirements.txt