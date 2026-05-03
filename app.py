import streamlit as st
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))
cm = pickle.load(open("cm.pkl", "rb"))

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #4fc3f7;
        text-align: center;
    }
    .stButton>button {
        background-color: #4fc3f7;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1f2937;
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🎬 Movie Review Sentiment Analyzer")

st.write("🚀 Fast AI-based sentiment prediction system")

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("Navigation", ["🎬 Predict Review", "📊 Dashboard"])

# =====================================================
# 🎬 PREDICTION PAGE
# =====================================================
if menu == "🎬 Predict Review":

    st.subheader("Enter Movie Details")

    movie = st.text_input("🎥 Movie Name")
    review = st.text_area("✍ Enter Movie Review")

    if st.button("Predict Sentiment 🚀"):

        if review.strip() == "":
            st.warning("Please enter a review")
        else:
            cleaned = clean_text(review)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]

            st.markdown("---")
            st.subheader(f"🎬 Movie: {movie}")

            if prediction == "positive":
                st.markdown(
                    "<div class='result-box' style='color:#00ff99'>😊 Positive Review</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='result-box' style='color:#ff4b4b'>😞 Negative Review</div>",
                    unsafe_allow_html=True
                )

# =====================================================
# 📊 DASHBOARD PAGE
# =====================================================
elif menu == "📊 Dashboard":

    st.title("📊 Model Performance Dashboard")

    # ---------------- METRIC CARDS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="🎯 Accuracy", value=f"{accuracy:.2f}")

    with col2:
        st.metric(label="📚 Model", value="Logistic Regression")

    st.markdown("---")

    # ---------------- ACCURACY GRAPH ----------------
    st.subheader("📈 Accuracy Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Model Accuracy"], [accuracy])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.markdown("---")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("🧾 Confusion Matrix")

    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax2, cmap="Blues")

    st.pyplot(fig2)

