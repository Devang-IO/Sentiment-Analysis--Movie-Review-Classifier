
# Sentiment Analysis  Movie Review Classifier

> A compact, beginner-friendly machine learning project that classifies movie reviews as **Positive** or **Negative**.  
> Trained end-to-end in **Google Colab**, packaged into pickled artifacts, and presented through a simple **Streamlit** UI.

----------
# UI Screenshots

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1271485c-039e-4767-94e7-651732ef7c30" alt="UI Screenshot 1" width="640" />
      <p><em>App input screen</em></p>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3e6ac725-31fc-4161-b9cb-d9bba03782f9" alt="UI Screenshot 2" width="640" />
      <p><em>Prediction result screen</em></p>
    </td>
  </tr>
</table>


## One-line summary

Lightweight sentiment classifier (TF-IDF + Logistic Regression) built in Colab and wrapped in a Streamlit app for quick demos and testing.

----------

## Quick highlights

-   **End-to-end ML project**: data → cleaning → TF-IDF → model → UI
    
-   **Baseline performance**: ≈ **89.5%** accuracy on the test split (Logistic Regression + TF-IDF)
    
-   **Where it was built**: Google Colab (full notebook included)
    
-   **How to demo**: Run locally with Streamlit or open the Colab notebook and run cells
    

----------

## Features

-   Real-time single-review predictions via a clean Streamlit interface
    
-   Simple and reproducible training notebook (Colab) that documents each step
    
-   Saved model & vectorizer (`*.pkl`) so the UI does not require retraining
    
-   Minimal, beginner-friendly codebase that’s easy to extend
    

----------

## Tech stack

-   Python 3.x
    
-   scikit-learn (TF-IDF, Logistic Regression)
    
-   pandas / numpy (data handling)
    
-   Streamlit (UI)
    
-   Google Colab (training & reproducibility)
    
-   Optional: pyngrok / cloudflared for temporary public demo tunnels
    
----------

## Colab notebook

<a href="https://colab.research.google.com/drive/1udu0z2zrXZJm2SfaJrWO-jblWLri5kJx?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width="200" />
</a>



The notebook demonstrates everything in a reproducible order:

1.  Load dataset (IMDb or CSV)
    
2.  Basic cleaning (remove HTML, non-letters, collapse spaces, lowercasing)
    
3.  TF-IDF vectorization (with option to enable n-grams)
    
4.  Train/test split and model training (Logistic Regression)
    
5.  Evaluation (accuracy, precision/recall/f1)
    
6.  Save model & vectorizer as `.pkl` files
    
7.  (Optional) demo Streamlit from Colab using pyngrok/cloudflared

----------

## How to run

### Option A - Run the Colab notebook (recommended for reproducibility)

1.  Open the Colab notebook link above.
    
2.  Run cells step-by-step to train the model or load pre-saved artifacts.
    
3.  Optionally use pyngrok / cloudflared instructions in the notebook to temporarily expose the Streamlit UI from Colab.
    

### Option B - Run locally (recommended for demos)

1.  Clone the repo and place `sentiment_model.pkl` and `tfidf_vectorizer.pkl` in the root.
    
2.  Create a virtual environment and install dependencies:
    

```python -m venv venv source venv/bin/activate # or venv\Scripts\activate on Windows pip install -r requirements.txt```

3.  Run:
    

`streamlit run app.py` 

4.  Open `http://localhost:8501` in your browser.
     

----------

## Limitations & why the model sometimes misclassifies

-   This is a **binary** classifier (positive vs negative). There is **no neutral** label in the training set.
    
-   TF-IDF + Logistic Regression is a strong baseline, but it struggles with sarcasm, idioms, and subtle phrasing (e.g., “I wish I could get those two hours of my life back”).
    
-   For better handling of subtle language, consider fine-tuning or using pretrained transformers (BERT variants).
    

----------

## License

Add a license as you wish (MIT recommended for small personal projects). Example `LICENSE` file content:

`MIT License
Copyright (c) YEAR YOUR_NAME ...` 

----------

