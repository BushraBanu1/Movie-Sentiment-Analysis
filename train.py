import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data = pd.read_csv("IMDB Dataset.csv")

data['cleaned'] = data['review'].apply(clean_text)

X = data['cleaned']
y = data['sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- METRICS ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 🔹 NEW METRICS (ADDED)
precision = precision_score(y_test, y_pred, pos_label="positive")
recall = recall_score(y_test, y_pred, pos_label="positive")
f1 = f1_score(y_test, y_pred, pos_label="positive")

# ---------------- SAVE EVERYTHING ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(accuracy, open("accuracy.pkl", "wb"))
pickle.dump(cm, open("cm.pkl", "wb"))

# 🔹 SAVE NEW FILES
pickle.dump(precision, open("precision.pkl", "wb"))
pickle.dump(recall, open("recall.pkl", "wb"))
pickle.dump(f1, open("f1.pkl", "wb"))

print("Model + All Metrics saved ✔")