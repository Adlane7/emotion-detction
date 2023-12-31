# **emotion-detction- using KNN ** 

attribute to kaggle data set https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset 


This project leverages the power of the k-Nearest Neighbors (k-NN) algorithm to classify emotions effectively. k-NN is a simple yet robust machine learning approach that makes predictions based on the similarity of data points. In this case, we use it to classify emotions like Angry, Happy, Sad, and Surprise.

Achievement: With our k-NN model, we have reached a notable 61% accuracy in emotion classification. While not a milestone, it demonstrates the model's effectiveness in understanding and distinguishing basic human emotions. Explore the project to learn more about our implementation and how you can use it in your applications.


# **Result ** 

| Metric            | Value               |
|-------------------|---------------------|
| Accuracy          | 0.6167              |
| Precision (0)     | 0.49                |
| Recall (0)        | 0.55                |
| F1-Score (0)      | 0.52                |
| Support (0)       | 2933                |
| Precision (1)     | 0.68                |
| Recall (1)        | 0.68                |
| F1-Score (1)      | 0.68                |
| Support (1)       | 5428                |
| Precision (2)     | 0.57                |
| Recall (2)        | 0.53                |
| F1-Score (2)      | 0.55                |
| Support (2)       | 3698                |
| Precision (3)     | 0.69                |
| Recall (3)        | 0.68                |
| F1-Score (3)      | 0.69                |
| Support (3)       | 2354                |
| Macro Avg         | 0.61                |
| Weighted Avg      | 0.62                |
| Total Support     | 14413               |

# **OpenCv**

After successfully running the project, three important files are generated:

knn_classifier:

This variable stores a k-Nearest Neighbors (k-NN) classifier model. k-NN is a supervised machine learning algorithm used for classification tasks. It makes predictions based on the majority class among the k-nearest data points in the training dataset. By loading the k-NN model, you can make predictions on new data without needing to retrain the model.
scaler:

This variable stores a saved StandardScaler. A StandardScaler is a preprocessing technique widely used in machine learning to standardize or normalize features in the dataset. It ensures that the features have a mean of 0 and a standard deviation of 1. Loading the scaler is essential to apply the same data scaling used during training to new data for consistent and accurate predictions.

label_encoder:

This variable stores a LabelEncoder, which is used to transform categorical labels into numerical values. When dealing with categorical target labels (e.g., class labels), a LabelEncoder converts them into numerical values for model training. Loading the label encoder ensures correct encoding and decoding of labels during predictions and evaluations.

Running the Emotion Recognition Script
To utilize the emotion classification model and see it in action, follow these steps:

Ensure you have the required dependencies installed

Run the open-cv-recognition.py script:

python open-cv-recognition.py

This script will demonstrate the emotion recognition capabilities of the project. It uses the saved k-NN model, StandardScaler, and LabelEncoder to classify emotions in real-time. Enjoy exploring the project and its applications!

