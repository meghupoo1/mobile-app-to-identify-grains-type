import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import graycomatrix, graycoprops
import cv2


def extract_color_histogram(img_array, bins=16):
    """Extract HSV color histogram features (fast)."""
    # Convert RGB to HSV
    hsv_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    
    # Compute histograms for H, S, V (reduced bins for speed)
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    
    # Concatenate and normalize
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    hist = hist / (hist.sum() + 1e-6)
    return hist


def extract_edge_features(gray_img):
    """Extract edge and blur features with additional texture metrics."""
    gray_img = gray_img.astype(np.uint8)
    
    # Canny edge detection
    edges = cv2.Canny(gray_img, 100, 200)
    edge_density = edges.sum() / (gray_img.size * 255.0)
    
    # Laplacian variance (texture roughness)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    laplacian_var = float(laplacian.var())
    
    # Contours/sharpness
    contour_score = float(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0).std())
    
    # Additional texture: local binary pattern variance (simpler than contours)
    mean_intensity = float(gray_img.mean())
    std_intensity = float(gray_img.std())
    
    return np.array([edge_density, laplacian_var, contour_score, mean_intensity, std_intensity])


def extract_all_features(img_array):
    """Extract combined features: raw pixels + color histogram + edge features."""
    # Raw pixel features
    raw_features = img_array.flatten()
    
    # Color histogram
    color_hist = extract_color_histogram(img_array)
    
    # Edge and texture features from grayscale
    gray_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edge_feat = extract_edge_features(gray_img)
    
    # Combine all
    combined = np.concatenate([raw_features, color_hist, edge_feat])
    return combined


def load_dataset(dataset_dir, image_size=(128, 128)):
    X = []
    y = []
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    for grain_class in sorted(dataset_path.iterdir()):
        if grain_class.is_dir():
            for img_path in sorted(grain_class.iterdir()):
                if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    continue
                try:
                    img = Image.open(img_path)
                    # Keep color (RGB) as in dataset
                    img = img.convert('RGB')
                    img = img.resize(image_size)
                    img_array = np.array(img)
                    # Extract combined features
                    features = extract_all_features(img_array)
                    X.append(features)
                    y.append(grain_class.name)
                except Exception as exc:
                    print(f"Error loading image {img_path}: {exc}")

    X = np.array(X)
    y = np.array(y)
    return X, y


def train_model(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # Enhanced RandomForest with balanced hyperparameters for better grain classification
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        criterion='gini'
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Grain Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.close()

    return classifier, label_encoder


def predict_image(classifier, label_encoder, image_path, image_size=(128, 128)):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img)
    # Extract combined features
    features = extract_all_features(img_array)
    img_array_reshaped = features.reshape(1, -1)
    prediction = classifier.predict(img_array_reshaped)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Confidence if supported
    confidence = None
    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(img_array_reshaped)[0]
        confidence = float(proba[prediction[0]] * 100)

    return predicted_label, img, confidence


if __name__ == '__main__':
    dataset_path = r"E:\grains dataset"
    test_image = Path(r"E:\grains dataset\rice_validation\rice1.jpg")

    X, y = load_dataset(dataset_path, image_size=(128, 128))
    classifier, label_encoder = train_model(X, y)

    if test_image.exists():
        predicted_label, img, confidence = predict_image(classifier, label_encoder, test_image)
        print(f"\nPredicted Grain Type for {test_image.name}: {predicted_label}")
        if confidence is not None:
            print(f"Confidence: {confidence:.1f}%")

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Predicted Grain Type: {predicted_label}")
        plt.axis('off')
        plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        print("Prediction result saved as 'prediction_result.png'")
        plt.close()
    else:
        print(f"Test image not found: {test_image}")
