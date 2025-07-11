import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import logging

from twiesnClassifier import TWIESNClassifier

logging.basicConfig(filename="result.csv", filemode='w')


def load_data(filepath):
    x_train = np.load(os.path.join(filepath, "x_train.npy"), allow_pickle=True).astype(np.float32)
    y_train = np.load(os.path.join(filepath, "y_train.npy"), allow_pickle=True).astype(np.int32)
    x_test = np.load(os.path.join(filepath, "x_test.npy"), allow_pickle=True).astype(np.float32)
    y_test = np.load(os.path.join(filepath, "y_test.npy"), allow_pickle=True).astype(np.int32)
    
    return x_train, y_train, x_test, y_test



def main(mode='all', cross=1):

    X_train, X_test, y_train, y_test = load_data(f'./data/{mode}/cross_{cross}/')
    
    model = TWIESNClassifier(
        n_inputs=4,
        n_reservoir=150,
        spectral_radius=1.0,
        sparsity=0.9,
        noise=0.002,
        washout_period=6,
        logistic_C=2.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_train, y_train)
    print(f"\nModel Accuracy on Train Set: {accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Test result: Acc: {acc*100}%, Pre: {pre*100}%, Rec: {rec*100}%, F1: {f1*100}%')
    
    logging.info(f'result, {acc}, {pre}, {rec}, {f1}')



if __name__ == '__main__':
    mode = 'all'
    for i in range(1, 5):
        main(mode, i)