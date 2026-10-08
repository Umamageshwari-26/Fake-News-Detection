# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 2. Load Datasets
# =========================
fake = pd.read_csv(r"C:\Users\Umamageshwari\Downloads\archive (8)\Fake.csv")
true = pd.read_csv(r"C:\Users\Umamageshwari\Downloads\archive (8)\True.csv")

# Add labels
fake["label"] = "fake"
true["label"] = "real"

# Combine datasets
data = pd.concat([fake, true])

# Keep only needed columns
data = data[["text", "label"]]

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

print("Dataset loaded successfully!")
print(data.head())

# =========================
# 3. Text Cleaning Function
# =========================
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)

# =========================
# 4. Split Data
# =========================
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Convert Text → Numbers (TF-IDF)
# =========================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 6. Train Model (Logistic Regression)
# =========================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# =========================
# 7. Evaluate Model
# =========================
# =========================
# 7. Evaluate Model
# =========================
y_pred = model.predict(X_test_vec)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 7.1 Plot Confusion Matrix (Matplotlib)
# =========================
plt.figure()

plt.imshow(cm)
plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Labels (adjust if needed)
labels = ["fake", "real"]
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

# Add numbers inside boxes
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.colorbar()
plt.show()


# =========================
# 8. Prediction Function
# =========================
def predict_news(news):
    news = clean_text(news)
    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)
    return result[0]

# =========================
# 9. Test with User Input
# =========================
while True:
    news_input = input("\nEnter news (or type 'exit'): ")
    if news_input.lower() == "exit":
        break
    prediction = predict_news(news_input)
    print("🧠 Prediction:", prediction.upper())
    # =========================
