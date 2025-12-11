# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample
# from sklearn.metrics import classification_report
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping

# # --------------------------------------------------------
# # 1. LOAD DATASET
# # --------------------------------------------------------
# df = pd.read_csv(r"D:\NNDL proj\movie_reviews_dataset.csv")   # <-- your CSV
# df = df[["review_text", "sentiment"]]   # Ensure correct columns

# print("\nOriginal Distribution:")
# print(df["sentiment"].value_counts())

# # --------------------------------------------------------
# # 2. BALANCE DATASET
# # --------------------------------------------------------
# pos = df[df["sentiment"] == "positive"]
# neg = df[df["sentiment"] == "negative"]
# neu = df[df["sentiment"] == "neutral"]

# max_size = max(len(pos), len(neg), len(neu))

# pos_bal = resample(pos, replace=True, n_samples=max_size, random_state=42)
# neg_bal = resample(neg, replace=True, n_samples=max_size, random_state=42)
# neu_bal = resample(neu, replace=True, n_samples=max_size, random_state=42)

# df_balanced = pd.concat([pos_bal, neg_bal, neu_bal])

# print("\nBalanced Dataset:")
# print(df_balanced["sentiment"].value_counts())

# # --------------------------------------------------------
# # 3. ENCODE LABELS
# # --------------------------------------------------------
# X = df_balanced["review_text"]
# y = df_balanced["sentiment"]

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# y_onehot = to_categorical(y_encoded, num_classes=3)

# # --------------------------------------------------------
# # 4. TRAIN-TEST SPLIT
# # --------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_onehot, test_size=0.2, random_state=42
# )

# # --------------------------------------------------------
# # 5. TF-IDF TEXT VECTORIZATION
# # --------------------------------------------------------
# vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
# X_train_vec = vectorizer.fit_transform(X_train).toarray()
# X_test_vec = vectorizer.transform(X_test).toarray()

# # --------------------------------------------------------
# # 6. NEURAL NETWORK MODEL
# # --------------------------------------------------------
# model = Sequential([
#     Dense(300, activation='relu', input_dim=X_train_vec.shape[1]),
#     Dropout(0.5),

#     Dense(200, activation='relu'),
#     Dropout(0.3),

#     Dense(100, activation='relu'),
#     Dense(3, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     X_train_vec, y_train,
#     epochs=12,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
#     verbose=1
# )

# # --------------------------------------------------------
# # 7. EVALUATION
# # --------------------------------------------------------
# y_pred = np.argmax(model.predict(X_test_vec), axis=1)
# y_true = np.argmax(y_test, axis=1)

# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# # --------------------------------------------------------
# # 8. ACCURACY CURVE
# # --------------------------------------------------------
# plt.figure(figsize=(6, 4))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Train vs Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(["Train", "Validation"])
# plt.tight_layout()
# plt.show()

# # --------------------------------------------------------
# # 9. PREDICT NEW SENTENCES (PURE ML)
# # --------------------------------------------------------
# def ml_predict(text):
#     text_vec = vectorizer.transform([text]).toarray()
#     pred = np.argmax(model.predict(text_vec))
#     return label_encoder.classes_[pred]

# while True:
#     text = input("\nEnter a movie review (or EXIT): ")
#     if text.lower() == "exit":
        
#         break
#     print("Predicted Sentiment:", ml_predict(text))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------------
df = pd.read_csv(r"D:\NNDL proj\movie_reviews_dataset.csv")
df = df[["review_text", "sentiment"]]

print("\nOriginal Distribution:")
print(df["sentiment"].value_counts())

# --------------------------------------------------------
# 2. BALANCE DATASET
# --------------------------------------------------------
pos = df[df["sentiment"] == "positive"]
neg = df[df["sentiment"] == "negative"]
neu = df[df["sentiment"] == "neutral"]

max_size = max(len(pos), len(neg), len(neu))

pos_bal = resample(pos, replace=True, n_samples=max_size, random_state=42)
neg_bal = resample(neg, replace=True, n_samples=max_size, random_state=42)
neu_bal = resample(neu, replace=True, n_samples=max_size, random_state=42)

df_balanced = pd.concat([pos_bal, neg_bal, neu_bal])

print("\nBalanced Dataset:")
print(df_balanced["sentiment"].value_counts())

# --------------------------------------------------------
# 3. ENCODE LABELS
# --------------------------------------------------------
X = df_balanced["review_text"]
y = df_balanced["sentiment"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=3)

# --------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# --------------------------------------------------------
# 5. TF-IDF TEXT VECTORIZATION
# --------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# --------------------------------------------------------
# 6. NEURAL NETWORK MODEL
# --------------------------------------------------------
model = Sequential([
    Dense(300, activation='relu', input_dim=X_train_vec.shape[1]),
    Dropout(0.5),

    Dense(200, activation='relu'),
    Dropout(0.3),

    Dense(100, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_vec, y_train,
    epochs=12,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

# --------------------------------------------------------
# 7. EVALUATION
# --------------------------------------------------------
y_pred = np.argmax(model.predict(X_test_vec), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# --------------------------------------------------------
# 7.1 CONFUSION MATRIX
# --------------------------------------------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 8. ACCURACY CURVE
# --------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Train vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 9. PREDICT NEW SENTENCES
# --------------------------------------------------------
def ml_predict(text):
    text_vec = vectorizer.transform([text]).toarray()
    pred = np.argmax(model.predict(text_vec))
    return label_encoder.classes_[pred]

while True:
    text = input("\nEnter a movie review (or EXIT): ")
    if text.lower() == "exit":
        break
    print("Predicted Sentiment:", ml_predict(text))

