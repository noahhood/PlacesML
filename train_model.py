import os
import pandas as pd
import numpy as np
import sys
import json
from matplotlib import pyplot as plt
import folium
from google.protobuf.json_format import MessageToDict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
os.chdir(sys.path[0])

with open("places_grid_search_50_by_20_plus_60_by_30.json", "r") as f:
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
points = list((zip(X.latitude, X.longitude)))

# Center the map roughly at the average location
avg_lat = sum(p[0] for p in points) / len(points)
avg_lon = sum(p[1] for p in points) / len(points)

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
colors = ['black', 'blue', 'green', 'yellow', 'orange', 'red']

for price_level in range(6):
    idx = np.where(df['price_level'] == price_level)[0]
    for lat, lon in np.array(points)[idx]:
        folium.CircleMarker(
            location=[lat, lon],
            fill=True,
            radius=4,
            fill_opacity=1,
            color=colors[price_level]
        ).add_to(m)

# Create a legend as HTML
legend_html = '''
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 150px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     ">
     &nbsp;<b>Legend</b><br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:black"></i>&nbsp; Price Level 0 <br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:blue"></i>&nbsp; Price Level 1    <br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:green"></i>&nbsp; Price Level 2    <br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:yellow"></i>&nbsp; Price Level 3    <br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:orange"></i>&nbsp; Price Level 4    <br>
     &nbsp;<i class="fa fa-circle fa-1x" style="color:red"></i>&nbsp; Price Level 5    <br>
     </div>
     '''

# Add the legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Save to HTML
m.save("map.html")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Define the base model
forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

scoring = {
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy"
}

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Perform kfold cross validation (only used to verify before training full model)
results = cross_validate(
    forest_model,
    X_train,
    y_train_enc,
    cv=skf,
    scoring=scoring,
    return_train_score=False
)

for metric in scoring:
    print(metric, results[f"test_{metric}"].mean())

# Fit on the full training data
forest_model.fit(X_train, y_train_enc)

print("Final model trained on the full training dataset.")

# Encode test labels if needed
y_test_enc = le.transform(y_test)  # use the same encoder as training

# Predict
y_pred = forest_model.predict(X_test)
y_pred_decoded = le.inverse_transform(y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred_decoded))
print("F1 weighted:", f1_score(y_test, y_pred_decoded, average="weighted"))
print("\nClassification report:\n", classification_report(y_test, y_pred_decoded))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Bins
bins = np.arange(0, 6) - 0.5
error_bins = np.arange(-2, 3) - 0.5

# Top subplot: True vs Predicted
ax1.hist(y_test, bins=bins, alpha=0.5, label='True', edgecolor='black')
ax1.hist(y_pred_decoded, bins=bins, alpha=0.5, label='Predicted', edgecolor='black')
ax1.set_xlabel('Price Level')
ax1.set_ylabel('Count')
ax1.set_title('Price Level: True vs Predicted')
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)
ax1.set_xlim(0.5, 5.5)

# Bottom subplot: Prediction Error
ax2.hist(y_test_enc - y_pred, bins=error_bins, color='red', alpha=0.7, edgecolor='black')
ax2.set_xlabel(r'$\Delta$ Price Level')
ax2.set_ylabel('Count')
ax2.set_title('Prediction Error (y_test - y_pred)')
ax2.grid(alpha=0.3)
ax2.set_xlim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("price_level_prediction_random_forest.png", dpi=300)

print("Done!")