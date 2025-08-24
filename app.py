import streamlit as st
import pickle

#Load saved model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

#Streamlit UI
st.set_page_config(page_title="Movie review sentiment app", page_icon="🍿")

st.title("Movie review sentiment analyzer")
st.title("Type a movei review below and check if its positive and negative!")

#Text input
user_input =  st.text_area("Enter your review:","")

if st.button("Predict Sentiment"):
    if user_input.strip() !="":
        #transform and predict
        review_vec = vectorizer.transform([user_input])
        prediction = model.predict(review_vec)[0]

        if prediction == "positive":
            st.success("Positive Review🟢")
        else:
            st.error("Negative Review🔴")
    else:
        st.warning("Please enter a review first.⚠️")