from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import numpy as np
import pickle

from twiesn import TWIESN


class TWIESNClassifier:
    """
    A classifier using a TWIESN as a fixed feature extractor and a
    linear classifier (Logistic Regression) as a trainable readout.
    """
    def __init__(self, n_inputs, n_reservoir=200, spectral_radius=0.95, sparsity=0.9, noise=0.001, washout_period=50, logistic_C=1.0, random_state=None):
        self.washout_period = washout_period
        self.feature_extractor = TWIESN(n_inputs, n_reservoir, spectral_radius, sparsity, noise, random_state)
        self.classifier = LogisticRegression(C=logistic_C, random_state=random_state, max_iter=1000)

    
    def _harvest_features(self, X):
        """
        Transforms a list of input sequences into a feature matrix.
        Each sequence is converted to a single feature vector by averaging its reservoir states.
        
        Args:
            X (list of np.ndarray): List of input sequences.
            
        Returns:
            np.ndarray: Feature matrix of shape (n_sequences, n_reservoir).
        """
        features = []
        for seq in X:
            state = self.feature_extractor.generate_state(seq, self.washout_period)
            if len(state) == 0:
                features.append(np.zeros(self.feature_extractor.n_reservoir))
            else:
                features.append(np.mean(state, axis=0))
        
        return np.array(features)

    
    def fit(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train (list of np.ndarray): List of training sequences.
            y_train (array-like): Corresponding class labels for each sequence.
        """
        print("Harvesting features from training data...")
        feature_matrix = self._harvest_features(X_train)
        
        print("Training the classifier...")
        self.classifier.fit(feature_matrix, y_train)
        print("Training complete.")
        
        return self
    
    
    def predict(self, X_test):
        """
        Predict class labels for new sequences.
        """
        if not hasattr(self.classifier, "classes_"):
            raise NotFittedError("This TWIESNClassifier instance is not fitted yet.")
        
        feature_matrix = self._harvest_features(X_test)
        return self.classifier.predict(feature_matrix)
    
    
    def predict_proba(self, X_test):
        """
        Predict class labels for new sequences.
        """
        if not hasattr(self.classifier, "classes_"):
            raise NotFittedError("This TWIESNClassifier instance is not fitted yet.")
        
        feature_matrix = self._harvest_features(X_test)
        return self.classifier.predict_proba(feature_matrix)
    
    
    def score(self, X_test, y_test):
        """
        Return the mean accuracy on the given test data and labels.
        """
        feature_matrix = self._harvest_features(X_test)
        return self.classifier.score(feature_matrix, y_test)
    

    def save(self, filepath):
        """
        Saves the entire trained model to a file using pickle.
        
        Args:
            filepath (str): The path to the file where the model will be saved.
        """
        print(f"Saving model to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print("Model saved successfully.")
        
        
    @classmethod
    def load(cls, filepath):
        """
        Loads a trained model from a file.
        
        This is a class method, so you can call it directly on the class:
        `loaded_model = TWIESNClassifier.load('path/to/model.pkl')`
        
        Args:
            filepath (str): The path to the saved model file.
            
        Returns:
            TWIESNClassifier: The loaded model instance.
            
        Note:
            Loading a pickled file can execute arbitrary code. Only load
            files from trusted sources.
        """
        print(f"Loading model from {filepath}...")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model
        