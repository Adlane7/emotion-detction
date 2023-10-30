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

After successfully running the code 3 file 

knn_classifier: This variable stores a k-Nearest Neighbors (k-NN) classifier model. A k-NN classifier is a supervised machine learning algorithm used for classification tasks. It makes predictions based on the majority class among the k-nearest data points in the training dataset. By loading the k-NN model, you can use it to make predictions on new data without needing to retrain the model.

scaler: This variable stores a saved StandardScaler. A StandardScaler is a preprocessing technique commonly used in machine learning to standardize or normalize features in the dataset. It ensures that the features have a mean of 0 and a standard deviation of 1. Loading the scaler is important because it allows you to apply the same data scaling that was used during training to new data for consistent and accurate predictions.

label_encoder: This variable stores a LabelEncoder. As mentioned in a previous response, a LabelEncoder is used to transform categorical labels into numerical values. When you have categorical target labels (e.g., class labels), you use a LabelEncoder to convert them into numerical values for training a machine learning model. When loading the label encoder, it ensures that you can correctly encode and decode the labels during predictions and evaluations.

after that you can run open-cv-recognition.py for running the script