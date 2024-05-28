import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
train_set_path = 'https://raw.githubusercontent.com/vinny380/bayes_network_genre_prediction/main/train.csv'
df = pd.read_csv(train_set_path)

# Distribution of Target Classes
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Song Genres')
plt.xlabel('Genre Class')
plt.ylabel('Frequency')
plt.show()

# Feature Distributions
features = ['Popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
plt.figure(figsize=(15, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Preprocessing the dataset
data_preprocessed = df.drop(['Artist Name', 'Track Name'], axis=1)
for col in data_preprocessed.columns:
    if data_preprocessed[col].dtype == 'float64':
        data_preprocessed[col].fillna(data_preprocessed[col].median(), inplace=True)
data_preprocessed['Duration_minutes'] = data_preprocessed['duration_in min/ms'] / 60000
data_preprocessed.drop(['duration_in min/ms'], axis=1, inplace=True)

# Identifying features to transform based on their skewness
skewed_features = data_preprocessed[features].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_features = skewed_features[skewed_features > 1]  # Arbitrary threshold for skewness

# Applying Log-transform to the skewed features
for feature in skewed_features.index:
    data_preprocessed[f'{feature}_log'] = np.log1p(data_preprocessed[feature])  # log1p for log(1+x) to handle zero values

# Updating the names of feature in the list after transformations
features_log_transformed = [f'{feature}_log' if feature in skewed_features.index else feature for feature in features]

# Plotting the distribution of log-transformed features
plt.figure(figsize=(15, 15))
for i, feature in enumerate(features_log_transformed, 1):
    plt.subplot(4, 4, i)
    sns.histplot(data_preprocessed[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

X_raw = data_preprocessed.drop(['Class'], axis=1).values
y_raw = data_preprocessed['Class'].values

# Adding seed for reproduciblity
np.random.seed(42)
indices = np.random.permutation(X_raw.shape[0])
train_idx, test_idx = indices[:int(X_raw.shape[0] * 0.8)], indices[int(X_raw.shape[0] * 0.8):]
X_train, X_test = X_raw[train_idx], X_raw[test_idx]
y_train, y_test = y_raw[train_idx], y_raw[test_idx]

# Computing the prior probability of each class, mean, and variance for each feature per class
unique_classes = np.unique(y_train)
prior_probabilities = np.array([len(y_train[y_train == c]) / len(y_train) for c in unique_classes])
class_feature_means = np.array([X_train[y_train == c].mean(axis=0) for c in unique_classes])
class_feature_variances = np.array([X_train[y_train == c].var(axis=0) for c in unique_classes])

def gaussian_pdf(x, mean, var):
    """Normal Distribution"""
    eps = 1e-6
    coeff = 1 / np.sqrt(2 * np.pi * var + eps)
    exponent = np.exp(-(x - mean)**2 / (2 * var + eps))
    return coeff * exponent

def predict(X, unique_classes, prior_probabilities, class_feature_means, class_feature_variances):
    """
    Predicts the class labels for the given input data using Naive Bayes Classifier.

    Parameters:
        X (numpy.ndarray): Input data features.
        unique_classes (numpy.ndarray): Array containing unique class labels.
        prior_probabilities (numpy.ndarray): Prior probabilities of each class.
        class_feature_means (numpy.ndarray): Mean values of features for each class.
        class_feature_variances (numpy.ndarray): Variance of features for each class.

    Returns:
        numpy.ndarray: Predicted class labels.
    """
    posteriors = []  # List to store posterior probabilities for each class
    for i, c in enumerate(unique_classes):
        prior = np.log(prior_probabilities[i])  # Log prior probability of the class
        # Calculate likelihoods for each feature given the class using Gaussian PDF
        likelihoods = gaussian_pdf(X, class_feature_means[i], class_feature_variances[i])
        # Compute log likelihood of the data
        log_likelihood = np.sum(np.log(likelihoods), axis=1)
        # Calculate posterior probability by adding log prior and log likelihood
        posterior = prior + log_likelihood
        posteriors.append(posterior)  # Append posterior for current class
    # Stack posteriors to form a matrix where each row represents the posteriors for a data point
    posteriors = np.column_stack(posteriors)
    # Predict the class label with the maximum posterior probability for each data point
    predictions = np.argmax(posteriors, axis=1)
    return predictions


# Callint the predict function
y_pred_custom = predict(X_test, unique_classes, prior_probabilities, class_feature_means, class_feature_variances)

# Calculating the accuracy
accuracy_custom = np.mean(y_pred_custom == y_test)
print(f"Custom Naive Bayes Classifier Accuracy: {accuracy_custom}")


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# Calculating Precision, Recall, and F1-Score using sicikit-learn
precision = precision_score(y_test, y_pred_custom, average='weighted')
recall = recall_score(y_test, y_pred_custom, average='weighted')
f1 = f1_score(y_test, y_pred_custom, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_custom)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show();