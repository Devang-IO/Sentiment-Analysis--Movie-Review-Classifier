import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Movie review sentiment app", page_icon="🍿")

# Load saved model and vectorizer
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

model = load_pickle("sentiment_model.pkl")
vectorizer = load_pickle("tfidf_vectorizer.pkl")

# UI
st.title("Movie Review Sentiment Analyzer 🍿")
st.write("Type a movie review below and check if it's positive or negative!")

user_input = st.text_area("Enter your review:", "")

def get_feature_names(vect):
    # sklearn >=1.0 uses get_feature_names_out
    try:
        return vect.get_feature_names_out()
    except:
        return vect.get_feature_names()

def explain_prediction(model, vectorizer, vec):
    """Return two lists: top positive contributing (word, score) and top negative contributing (word, score)."""
    # Only works for linear models with coef_
    if not hasattr(model, "coef_"):
        return None, None

    # Get feature names
    feature_names = get_feature_names(vectorizer)
    coef = model.coef_
    # For binary classification sklearn stores coef_ shape as (1, n_features)
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    elif coef.ndim == 2 and coef.shape[0] > 1:
        # multiclass: choose coef for the predicted positive class later; for now use first row
        coef = coef[0]

    # Obtain tfidf values for this sample
    try:
        tfidf_values = vec.toarray()[0]
    except:
        tfidf_values = vec.tocsr().toarray()[0]

    # Contribution per feature = coef * tfidf_value
    contributions = coef * tfidf_values

    # Get top positive contributions (supporting positive sentiment)
    top_pos_idx = np.argsort(contributions)[-10:][::-1]  # descending
    top_neg_idx = np.argsort(contributions)[:10]        # ascending (most negative)

    top_pos = [(feature_names[i], float(contributions[i])) for i in top_pos_idx if tfidf_values[i] != 0]
    top_neg = [(feature_names[i], float(contributions[i])) for i in top_neg_idx if tfidf_values[i] != 0]

    return top_pos, top_neg

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.⚠️")
    else:
        # transform and predict
        review_vec = vectorizer.transform([user_input])
        try:
            prediction = model.predict(review_vec)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Probability / confidence
        prob = None
        try:
            proba = model.predict_proba(review_vec)[0]  # array of probabilities
            # get index of predicted class
            classes = list(model.classes_)
            pred_index = classes.index(prediction) if prediction in classes else None
            if pred_index is not None:
                prob = proba[pred_index]
        except Exception:
            proba = None
            prob = None

        # Show results
        if prediction == "positive":
            if prob:
                st.success(f"Positive Review 🟢 — Confidence: {prob:.2%}")
                st.progress(min(int(prob * 100), 100))
            else:
                st.success("Positive Review 🟢")
        else:
            if prob:
                st.error(f"Negative Review 🔴 — Confidence: {prob:.2%}")
                st.progress(min(int(prob * 100), 100))
            else:
                st.error("Negative Review 🔴")

        # Explainability: show top contributing words
        top_pos, top_neg = explain_prediction(model, vectorizer, review_vec)
        if top_pos is None and top_neg is None:
            st.info("No explainability available for this model (requires linear model with `coef_`).")
        else:
            # Show top positive contributors
            if len(top_pos) > 0:
                st.markdown("**Top words supporting positive sentiment (contribution scores):**")
                df_pos = pd.DataFrame(top_pos, columns=["word", "contribution"])
                # show only top 5
                st.table(df_pos.head(5).assign(contribution=lambda x: x["contribution"].map(lambda v: f"{v:.4f}")))
                st.bar_chart(df_pos.head(5).set_index("word")["contribution"])
            else:
                st.write("No strong positive-contributing words found in the review.")

            # Show top negative contributors
            if len(top_neg) > 0:
                st.markdown("**Top words supporting negative sentiment (contribution scores):**")
                df_neg = pd.DataFrame(top_neg, columns=["word", "contribution"])
                st.table(df_neg.head(5).assign(contribution=lambda x: x["contribution"].map(lambda v: f"{v:.4f}")))
                st.bar_chart(df_neg.head(5).set_index("word")["contribution"])
            else:
                st.write("No strong negative-contributing words found in the review.")

        # Optionally show raw probabilities for both classes
        if proba is not None:
            prob_df = pd.DataFrame({
                "class": list(model.classes_),
                "probability": [float(p) for p in proba]
            }).sort_values("probability", ascending=False)
            st.markdown("**Class probabilities:**")
            st.table(prob_df.style.format({"probability": "{:.2%}"}))
