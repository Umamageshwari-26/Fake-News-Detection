
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


fake = pd.read_csv(r"C:\Users\Umamageshwari\Downloads\archive (8)\Fake.csv")
true = pd.read_csv(r"C:\Users\Umamageshwari\Downloads\archive (8)\True.csv")


fake["label"] = "fake"
true["label"] = "real"


data = pd.concat([fake, true])


data = data[["text", "label"]]


data = data.sample(frac=1).reset_index(drop=True)

print("Dataset loaded successfully!")
print(data.head())


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)


X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


plt.figure()

plt.imshow(cm)
plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


labels = ["fake", "real"]
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)


for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.colorbar()
plt.show()



def predict_news(news):
    news = clean_text(news)
    news_vec = vectorizer.transform([news])
    result = model.predict(news_vec)
    return result[0]


while True:
    news_input = input("\nEnter news (or type 'exit'): ")
    if news_input.lower() == "exit":
        break
    prediction = predict_news(news_input)
    print("🧠 Prediction:", prediction.upper())
   
