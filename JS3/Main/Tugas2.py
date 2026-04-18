# Tugas Nomor 2 Data Mining - Regression Logistic Model
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('Dataset/framingham.csv')

print("Missing Values:\n", df.isnull().sum())
df.fillna(df.median(), inplace=True)

X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
cm= (confusion_matrix(y_test, y_pred))
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

plt.figure()
sns.countplot(x=y)
plt.title("Distribusi Target (TenYearCHD)")
plt.xlabel("Kelas (0 = Tidak Sakit, 1 = Sakit)")
plt.ylabel("Jumlah")
plt.show()