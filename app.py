import streamlit as st
import tensorflow as tf
import pickle
import re

# Load ANN model
nn = tf.keras.models.load_model("restaurant_sentiment.keras")

# Load CountVectorizer
with open("cv.pkl", "rb") as file:
    cv = pickle.load(file)


# Function for cleaning review
def clean_review(review):

    review = review.lower()

    review = re.sub("[^a-zA-Z]", " ", review)

    return review


# Streamlit UI
st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write("Enter a restaurant review to check whether it is positive or negative.")

review = st.text_area(
    "Enter Restaurant Review",
    placeholder="Example: The food was amazing and the service was excellent."
)


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Clean review
        cleaned_review = clean_review(review)

        # Convert text into numbers
        data_pred = cv.transform([cleaned_review])

        # ANN prediction
        prediction = nn.predict(data_pred.toarray())

        probability = prediction[0][0]

        # Convert probability into 0/1
        if probability > 0.5:

            st.success("😊 Positive Review")

            st.write(
                "Positive probability:",
                round(float(probability) * 100, 2),
                "%"
            )

        else:

            st.error("😞 Negative Review")

            st.write(
                "Negative probability:",
                round((1 - float(probability)) * 100, 2),
                "%"
            )