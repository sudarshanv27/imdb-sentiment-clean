"""
IMDb Sentiment Analysis
Streamlit Application

Uses:
    TF-IDF Vectorizer
    Logistic Regression

Final test accuracy:
    89.43%
"""

import streamlit as st

from src.predict import load_models, predict_sentiment


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def get_models():
    """
    Load the trained ML models once and reuse them.
    """
    return load_models()


vectorizer, model = get_models()


# ============================================================
# HEADER
# ============================================================

st.title("🎬 IMDb Sentiment Analyzer")

st.write(
    "Enter a movie review below and the trained machine "
    "learning model will predict whether the sentiment is "
    "positive or negative."
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ About the Model"):

    st.write(
        """
        **Model:** Logistic Regression

        **Feature Extraction:** TF-IDF

        **Number of Features:** 10,000

        **Training Reviews:** 20,000

        **Validation Reviews:** 5,000

        **Test Reviews:** 25,000

        **Final Test Accuracy:** 89.43%

        The model was trained using the IMDb movie review
        dataset containing 50,000 labeled reviews.
        """
    )


# ============================================================
# REVIEW INPUT
# ============================================================

st.subheader("📝 Enter a Movie Review")

review = st.text_area(
    "Movie Review",
    placeholder=(
        "Example: This movie was absolutely fantastic. "
        "The acting was excellent and I loved the story."
    ),
    height=180,
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True,
):

    # Check empty input
    if not review.strip():

        st.warning(
            "Please enter a movie review before analyzing."
        )

    else:

        try:

            # Make prediction
            result = predict_sentiment(
                review,
                vectorizer,
                model,
            )

            sentiment = result["sentiment"]
            confidence = result["confidence"]

            positive_probability = (
                result["positive_probability"]
            )

            negative_probability = (
                result["negative_probability"]
            )


            # ====================================================
            # DISPLAY RESULT
            # ====================================================

            st.divider()

            st.subheader("📊 Prediction")

            if sentiment == "POSITIVE":

                st.success(
                    f"🟢 POSITIVE — {confidence:.2f}% confidence"
                )

            else:

                st.error(
                    f"🔴 NEGATIVE — {confidence:.2f}% confidence"
                )


            # ====================================================
            # PROBABILITY METRICS
            # ====================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Positive Probability",
                    f"{positive_probability:.2f}%",
                )

            with col2:

                st.metric(
                    "Negative Probability",
                    f"{negative_probability:.2f}%",
                )


            # ====================================================
            # PROBABILITY BAR
            # ====================================================

            st.write("### Confidence")

            st.progress(
                int(round(confidence))
            )


            # ====================================================
            # REVIEW SUMMARY
            # ====================================================

            with st.expander("🔎 Review Analyzed"):

                st.write(review)


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IMDb Sentiment Analysis | "
    "TF-IDF + Logistic Regression | "
    "Test Accuracy: 89.43%"
)