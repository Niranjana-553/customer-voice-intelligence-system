

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

app = Flask(__name__)



with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

print("Model and vectorizer loaded successfully!")




lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

negation_words = {
    "no", "not", "nor", "never", "neither", "nothing",
    "nowhere", "nobody", "hardly", "barely", "scarcely",
    "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't",
    "couldn't", "won't", "can't"
}
stop_words -= negation_words


def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens if t not in stop_words]
    return " ".join(tokens)




@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    review     = None
    error      = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if not review:
            error = "Please enter a review before analysing."
        else:
            cleaned    = preprocess(review)
         
            vectorized = tfidf.transform([cleaned])

            pred = svm_model.predict(vectorized)[0]
            probs = svm_model.predict_proba(vectorized)[0]

            prob_negative = round(float(probs[0]) * 100, 1)
            prob_positive = round(float(probs[1]) * 100, 1)

            prediction = "Recommended" if pred == 1 else "Not Recommended"
            confidence = {
                "Recommended"    : prob_positive,
                "Not Recommended": prob_negative,
            }

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           review=review,
                           error=error)


if __name__ == "__main__":
    app.run(debug=False)
