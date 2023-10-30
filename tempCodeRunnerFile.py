knn_classifier = joblib.load('knn_model.pkl')  # Load your saved k-NN model
scaler = joblib.load('scaler.pkl')  # Load your saved StandardScaler
label_encoder = joblib.load('label_encoder.pkl')  # Load your saved LabelEncoder