# Grain Type Identifier

This project trains a Naive Bayes classifier on a grain image dataset and predicts grain type from a sample image.

## Setup

1. Activate the virtual environment:
   ```powershell
   & "E:\ml project\classproject\Scripts\activate"
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the application:
   ```powershell
   python app.py
   ```

## Dataset

The dataset is located at `E:\grains dataset` and was extracted from `E:\grains dataset.zip`.

## Streamlit App

To run the Streamlit upload app:

```powershell
streamlit run "E:\ml project\streamlit_app.py"
```

## Notes

- Images are resized to `64x64` and converted to grayscale.
- `GaussianNB` is used for classification.
- A confusion matrix and classification report are displayed.


## team members name 
meghashree KUB24CSE648
poornima   KUB24CSE661
shwetha    KUB24CSE681