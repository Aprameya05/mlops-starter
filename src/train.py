import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime

# Load dataset
data = pd.read_csv("data/data.csv")
X = data[["x"]]
y = data["y"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Evaluate model
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

# Save model
os.makedirs("models", exist_ok=True)
model_path = f"models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model, model_path)

# Log experiment
with open("experiments/log.txt", "a") as f:
    f.write(f"{model_path}, MSE={mse}\n")

print("Training complete. Model saved.")
