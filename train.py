import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="insta"
)

df = pd.read_sql("SELECT * FROM data", conn)


df["engagement_ratio"] = df["followers"] / (df["follows"] + 1)
df["posts_per_follower"] = df["posts"] / (df["followers"] + 1)


X = df[["posts", "followers", "follows", "engagement_ratio", "posts_per_follower"]]
y = df["fake"]


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model & Scaler Saved")