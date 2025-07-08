import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import numpy as np

from twiesnClassifier import TWIESNClassifier


def generate_3d_gesture_data(n_samples_per_class=20, noise_level=0.05):
    """
    Generates a synthetic 3D gesture dataset with time-warping.
    - Class 0: Circle in XY plane
    - Class 1: Helix along Z axis
    - Class 2: Figure-8 in XY plane
    """
    X, y = [], []
    
    # Define gesture generation functions
    def get_circle(t):
        return np.array([np.cos(t), np.sin(t), np.zeros_like(t)]).T
        
    def get_helix(t):
        return np.array([np.cos(t), np.sin(t), t / (2 * np.pi)]).T
        
    def get_figure8(t):
        return np.array([np.sin(t), np.cos(t/2), np.zeros_like(t)]).T
    
    gestures = {0: get_circle, 1: get_helix, 2: get_figure8}
    
    for class_id, func in gestures.items():
        for _ in range(n_samples_per_class):
            length = np.random.randint(80, 200)
            end_time = np.random.uniform(1.8, 2.2) * (2 * np.pi)
            t = np.linspace(0, end_time, length)

            clean_gesture = func(t)
            noise = np.random.randn(*clean_gesture.shape) * noise_level
            noise_gesture = clean_gesture + noise
            
            X.append(noise_gesture)
            y.append(class_id)
    
    return X, np.array(y)


def main():
    X, y = generate_3d_gesture_data(50, 0.03)

    # fig = plt.figure(figsize=(12, 4))
    class_ids = np.unique(y)
    # for i, class_id in enumerate(class_ids):
    #     ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    #     sample_idx = np.where(y==class_id)[0][0]
    #     sample = X[sample_idx]
    #     ax.plot(sample[:, 0], sample[:, 1], sample[:, 2])
    #     ax.set_title(f"Sample from Class {class_id}")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     plt.suptitle("3D Gesture Samples", fontsize=16)
    #     plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    #     plt.show()
       
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model = TWIESNClassifier(
        n_inputs=3,
        n_reservoir=350,
        spectral_radius=1.1,
        sparsity=0.7,
        noise=0.002,
        washout_period=20,
        logistic_C=10.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_ids)
    plt.show()



if __name__ == '__main__':
    main()