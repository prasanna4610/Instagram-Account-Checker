from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Input schema
class UserInput(BaseModel):
    posts: int
    followers: int
    follows: int

# Home route
@app.get("/")
def home():
    return {"message": "Backend is running"}

# Prediction route
@app.post("/check")
def check_account(data: UserInput):

    # Feature engineering
    engagement_ratio = data.followers / (data.follows + 1)
    posts_per_follower = data.posts / (data.followers + 1)

    features = np.array([[
        data.posts,
        data.followers,
        data.follows,
        engagement_ratio,
        posts_per_follower
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    if prediction == 0:
        result = "Real Account"
    else:
        result = "Fake Account"

    return {
        "result": result
    }
# Load ML model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Input format
class UserInput(BaseModel):
    posts: int
    followers: int
    follows: int

# Home route
@app.get("/")
def home():
    return {"message": "Backend is running"}

# Prediction API
@app.post("/check")
def check(data: UserInput):

    engagement_ratio = data.followers / (data.follows + 1)
    posts_per_follower = data.posts / (data.followers + 1)

    features = np.array([[
        data.posts,
        data.followers,
        data.follows,
        engagement_ratio,
        posts_per_follower
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    result = "Real Account" if prediction == 0 else "Fake Account"

    return {
    #    "score": int(prediction),
        "result": result
    }
