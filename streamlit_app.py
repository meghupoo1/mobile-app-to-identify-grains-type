import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import cv2


def extract_color_histogram(img_array, bins=16):
    """Extract HSV color histogram features (fast)."""
    hsv_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    hist = hist / (hist.sum() + 1e-6)
    return hist


def extract_edge_features(gray_img):
    """Extract edge and blur features (fast alternative to GLCM)."""
    gray_img = gray_img.astype(np.uint8)
    edges = cv2.Canny(gray_img, 100, 200)
    edge_density = edges.sum() / (gray_img.size * 255.0)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    laplacian_var = float(laplacian.var())
    contour_score = float(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0).std())
    return np.array([edge_density, laplacian_var, contour_score])


def extract_all_features(img_array):
    """Extract combined features: raw pixels + color histogram + edge features."""
    raw_features = img_array.flatten()
    color_hist = extract_color_histogram(img_array)
    gray_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edge_feat = extract_edge_features(gray_img)
    combined = np.concatenate([raw_features, color_hist, edge_feat])
    return combined


@st.cache_data(show_spinner=False)
def load_dataset(dataset_dir, image_size=(64, 64)):
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
                    img = img.convert('RGB')
                    img = img.resize(image_size)
                    img_array = np.array(img)
                    features = extract_all_features(img_array)
                    X.append(features)
                    y.append(grain_class.name)
                except Exception:
                    continue

    return np.array(X), np.array(y)


@st.cache_data(show_spinner=False)
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

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return classifier, label_encoder, accuracy, cm, X_test, y_test, y_pred


def preprocess_image(image_bytes, image_size=(64, 64)):
    img = Image.open(image_bytes)
    img = img.convert('RGB')
    img = img.resize(image_size)
    return img


def pretty_label(label):
    return label.replace('_', ' ').title()


def predict_image(classifier, label_encoder, img, image_size=(64, 64)):
    """Predict using the exact same feature extraction as training."""
    img_array = np.array(img)
    features = extract_all_features(img_array).reshape(1, -1)

    prediction = classifier.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(features)[0]
        confidence = float(proba[prediction[0]] * 100)
    else:
        confidence = None

    return predicted_label, confidence



def render_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title='Grain Classifier', layout='wide')
    st.title('Grain Type Identifier')
    st.write('Upload a grain image and the model will predict the grain type.')

    dataset_path = st.sidebar.text_input('Dataset path', value=r'E:\grains dataset')
    st.sidebar.markdown('---')
    st.sidebar.write('Upload an image file to classify:')

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    st.sidebar.write('Model: RandomForest')
    st.sidebar.write('Image size: 64x64 color (RGB)')

    if not Path(dataset_path).exists():
        st.error(f'Dataset folder not found: {dataset_path}')
        return

    with st.spinner('Loading dataset and training model...'):
        X, y = load_dataset(dataset_path)
        classifier, label_encoder, accuracy, cm, X_test, y_test, y_pred = train_model(X, y)

    st.success('Model trained successfully.')
    st.metric('Validation accuracy', f'{accuracy:.4f}')

    st.subheader('Prediction')
    if uploaded_file is not None:
        try:
            image = preprocess_image(uploaded_file)
            predicted_label, confidence = predict_image(classifier, label_encoder, image)
            st.image(image, caption='Uploaded image', width=250)
            st.write('**Predicted grain type:**', pretty_label(predicted_label))
            if confidence is not None:
                st.write('**Confidence:**', f'{confidence:.1f}%')
        except Exception as exc:
            st.error(f'Unable to process image: {exc}')
    else:
        st.info('Upload a grain image to get a prediction.')

    with st.expander('Training details'):
        st.write('Number of samples:', X.shape[0])
        st.write('Number of classes:', len(label_encoder.classes_))
        st.write('Classes:', list(label_encoder.classes_))

        st.write('Confusion Matrix')
        fig = render_confusion_matrix(cm, label_encoder.classes_)
        st.pyplot(fig)

        st.write('Classification Report')
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        st.dataframe(report)

    st.sidebar.markdown('---')
    st.sidebar.write('Tip: use a clean image and simple background for best results.')


if __name__ == '__main__':
    main()
