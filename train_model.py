import os
import pandas as pd
import numpy as np
import sys
import json
from matplotlib import pyplot as plt
import folium
from google.protobuf.json_format import MessageToDict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
os.chdir(sys.path[0])

with open("places_grid_search_50_by_20.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame([{
    "latitude": item["location"]["latitude"],
    "longitude": item["location"]["longitude"],
    "price_level": item["price_level"],
    "primary_type": item["primary_type"],
    "rating": item["rating"],
    "user_rating_count": item["user_rating_count"],
} for item in data if "price_level" in item and "rating" in item and "user_rating_count" in item])

X = df[["latitude", "longitude", "rating", "user_rating_count"]]
y = df["price_level"]

# Example coordinates
points = list(zip(X.latitude, X.longitude))

# Center the map roughly at the average location
avg_lat = sum(p[0] for p in points) / len(points)
avg_lon = sum(p[1] for p in points) / len(points)

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)

for lat, lon in points:
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        fill=True
    ).add_to(m)

# Save to HTML
m.save("map.html")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Bins
bins = np.arange(0, 6) - 0.5
error_bins = np.arange(-2, 3) - 0.5

# Top subplot: True vs Predicted
ax1.hist(y_test, bins=bins, alpha=0.5, label='True', edgecolor='black')
ax1.hist(y_pred, bins=bins, alpha=0.5, label='Predicted', edgecolor='black')
ax1.set_ylabel('Count')
ax1.set_title('Price Level: True vs Predicted')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)
ax1.set_xlim(0.5, 5.5)

# Bottom subplot: Prediction Error
ax2.hist(y_test - y_pred, bins=error_bins, color='red', alpha=0.7, edgecolor='black')
ax2.set_xlabel(r'$\Delta$ Price Level ')
ax2.set_ylabel('Count')
ax2.set_title('Prediction Error (y_test - y_pred)')
ax2.grid(alpha=0.3)
ax2.set_xlim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("price_level_prediction_random_forest.png", dpi=300)

print("Done!")