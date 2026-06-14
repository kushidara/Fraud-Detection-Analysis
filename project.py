import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

print("New Shape:", df.shape)
print(df['Class'].value_counts())
fraud_percentage = df['Class'].value_counts(normalize=True) * 100
print(fraud_percentage)
import matplotlib.pyplot as plt

df['Class'].value_counts().plot(kind='bar')

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
print(df['Amount'].describe())
def risk_level(amount):
    if amount > 1000:
        return "High"
    elif amount > 500:
        return "Medium"
    else:
        return "Low"

df['Risk_Level'] = df['Amount'].apply(risk_level)
print(df['Risk_Level'].value_counts())
fraud = df[df['Class'] == 1]

print(fraud.shape)
print(fraud['Amount'].describe())
print(fraud['Risk_Level'].value_counts())
import matplotlib.pyplot as plt

fraud['Risk_Level'].value_counts().plot(kind='bar')

plt.title("Fraud Transactions by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Number of Frauds")
plt.show()
X = df.drop(['Class', 'Risk_Level'], axis=1)

y = df['Class']

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Trained Successfully")
predictions = model.predict(X_test)

print(predictions[:20])
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance.head(10))
import matplotlib.pyplot as plt

feature_importance.head(10).plot(
    x='Feature',
    y='Importance',
    kind='bar'
)

plt.title("Top 10 Important Features")
plt.show()
feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)
