import streamlit as st
import pickle
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="Movie review sentiment app", page_icon="🍿", layout="centered")

# -------------------------
# Helpers: load artifacts
# -------------------------
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()

model = load_pickle("sentiment_model.pkl")
vectorizer = load_pickle("tfidf_vectorizer.pkl")

# -------------------------
# Helpers: feature names & explain
# -------------------------
def get_feature_names(vect):
    try:
        return vect.get_feature_names_out()
    except:
        return vect.get_feature_names()

def explain_prediction(model, vectorizer, vec):
    """Return two lists: top positive contributing (word, score) and top negative contributing (word, score)."""
    if not hasattr(model, "coef_"):
        return None, None

    feature_names = get_feature_names(vectorizer)
    coef = model.coef_
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    elif coef.ndim == 2 and coef.shape[0] > 1:
        coef = coef[0]

    try:
        tfidf_values = vec.toarray()[0]
    except:
        tfidf_values = vec.tocsr().toarray()[0]

    contributions = coef * tfidf_values
    top_pos_idx = np.argsort(contributions)[-10:][::-1]
    top_neg_idx = np.argsort(contributions)[:10]

    top_pos = [(feature_names[i], float(contributions[i])) for i in top_pos_idx if tfidf_values[i] != 0]
    top_neg = [(feature_names[i], float(contributions[i])) for i in top_neg_idx if tfidf_values[i] != 0]

    return top_pos, top_neg

# -------------------------
# Page UI
# -------------------------
st.title("Movie Review Sentiment Analyzer 🍿")
st.write("Type a movie review below and check if it's positive or negative — or upload a CSV to classify many reviews at once.")

# ---------- Single review ----------
st.header("Single review")
user_input = st.text_area("Enter your review:", "", key="single_input", height=140)

if st.button("Predict Sentiment", key="single_predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review first. ⚠️")
    else:
        review_vec = vectorizer.transform([user_input])
        try:
            prediction = model.predict(review_vec)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Probability
        prob = None
        proba = None
        try:
            proba = model.predict_proba(review_vec)[0]
            classes = list(model.classes_)
            pred_index = classes.index(prediction) if prediction in classes else None
            if pred_index is not None:
                prob = proba[pred_index]
        except Exception:
            proba = None
            prob = None

        # Show result
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

        # Explainability
        top_pos, top_neg = explain_prediction(model, vectorizer, review_vec)
        if top_pos is None and top_neg is None:
            st.info("Explainability not available (model lacks `coef_`).")
        else:
            if len(top_pos) > 0:
                st.markdown("**Top positive-contributing words:**")
                df_pos = pd.DataFrame(top_pos, columns=["word", "contribution"]).head(5)
                st.table(df_pos.assign(contribution=lambda x: x["contribution"].map(lambda v: f"{v:.4f}")))
            if len(top_neg) > 0:
                st.markdown("**Top negative-contributing words:**")
                df_neg = pd.DataFrame(top_neg, columns=["word", "contribution"]).head(5)
                st.table(df_neg.assign(contribution=lambda x: x["contribution"].map(lambda v: f"{v:.4f}")))

        if proba is not None:
            prob_df = pd.DataFrame({
                "class": list(model.classes_),
                "probability": [float(p) for p in proba]
            }).sort_values("probability", ascending=False)
            st.markdown("**Class probabilities:**")
            st.table(prob_df.style.format({"probability": "{:.2%}"}))

# ---------- Bulk CSV upload ----------
st.header("Batch upload (CSV)")
st.write("Upload a CSV containing reviews. Column name `review` is expected; otherwise the first text column will be used.")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="upload_csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Find the text column (prefer 'review' or 'text', else the first column)
    text_col = None
    candidates = [c for c in df.columns if c.lower() in ("review", "text", "comment", "sentence")]
    if candidates:
        text_col = candidates[0]
    else:
        # fallback to first column that contains strings
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break

    if text_col is None:
        st.error("No text column found in the uploaded CSV.")
    else:
        st.write(f"Using column: **{text_col}**")
        reviews = df[text_col].astype(str).tolist()

        # Transform and predict in one go
        with st.spinner("Vectorizing and predicting..."):
            X = vectorizer.transform(reviews)
            try:
                preds = model.predict(X)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            proba_arr = None
            try:
                proba_arr = model.predict_proba(X)  # shape (n, n_classes)
            except Exception:
                proba_arr = None

        # Build results dataframe
        results_df = df.copy()
        results_df["prediction"] = preds
        if proba_arr is not None:
            for idx, cls in enumerate(model.classes_):
                results_df[f"prob_{cls}"] = proba_arr[:, idx]

        # Show top rows and aggregate chart
        st.subheader("Predictions (first 200 rows)")
        st.dataframe(results_df.head(200))

        st.subheader("Aggregate summary")
        try:
            vc = results_df["prediction"].value_counts()
            st.bar_chart(vc)
            st.write(vc)
        except Exception:
            st.write("Could not compute aggregate chart.")

        # Download results CSV
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download predictions CSV", data=csv_bytes,
                           file_name="predictions.csv", mime="text/csv")

        # Interactive explainability for a chosen row
        st.subheader("Explain sample from uploaded CSV")
        max_idx = min(len(results_df) - 1, 9999)
        chosen = st.number_input(f"Pick row index (0 to {max_idx})", min_value=0, max_value=max_idx, value=0, step=1)
        sample_text = results_df.iloc[chosen][text_col]
        st.markdown(f"**Row {chosen} — review:**")
        st.write(sample_text)

        # Explain for chosen row
        sample_vec = vectorizer.transform([str(sample_text)])
        top_pos, top_neg = explain_prediction(model, vectorizer, sample_vec)
        if top_pos is None and top_neg is None:
            st.info("Explainability not available for this model (requires linear model with `coef_`).")
        else:
            if len(top_pos) > 0:
                st.markdown("**Top positive-contributing words:**")
                df_pos = pd.DataFrame(top_pos, columns=["word", "contribution"]).head(5)
                st.table(df_pos.assign(contribution=lambda x: x["contribution"].map(lambda v: f"{v:.4f}")))
            if len(top_neg) > 0:
                st.markdown("**Top negative-contributing words:**")
                df_neg = pd.DataFrame(top_neg, columns=["word", "contribution"]).head(5)
                st.table(df_neg.assign(contribution=lambda x: x["contribution"].map(lambda v: f"{v:.4f}")))
