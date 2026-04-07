from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Input schema
class UserInput(BaseModel):
    posts: int
    followers: int
    follows: int

# Serve HTML page
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# Prediction API
@app.post("/check")
def check_account(user: UserInput):
    engagement_ratio = user.followers / (user.follows + 1)
    posts_per_follower = user.posts / (user.followers + 1)

    features = np.array([[
        user.posts,
        user.followers,
        user.follows,
        engagement_ratio,
        posts_per_follower
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    result = "Real Account" if prediction == 0 else "Fake Account"

    return {"result": result}
