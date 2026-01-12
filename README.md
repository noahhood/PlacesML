# PlacesML: Restaurant Price Level Prediction with Machine Learning

This project predicts a restaurant’s **price level** using location and popularity signals from the **Google Places API**. The final model is a **Random Forest classifier** that achieves approximately **61% accuracy** on a held-out test set.

---

## Project Goal

Estimate a restaurant’s price level (`0–5`, as defined by Google Places) using minimal, widely available features. This is framed as a **multiclass classification** problem.

---

## Data Collection

Restaurant data is collected using the **Google Places Nearby Search API** by scanning a geographic grid and querying nearby restaurants. Results are deduplicated by Place ID.

Only restaurants with complete metadata are retained.

**Fields used:**
- Latitude
- Longitude
- Rating
- User rating count
- Price level (target)

Data is stored as JSON and converted into a Pandas DataFrame for modeling.

---

## Features and Target

**Input Features**
- `latitude`
- `longitude`
- `rating`
- `user_rating_count`

**Target Variable**
- `price_level` (categorical, integer)

---

## Model

A **RandomForestClassifier** from scikit-learn is used.

**Configuration**
- Trees: `100`
- Train / Test split: `80% / 20%`
- Random state: `42`

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

## Results

**Accuracy:** 61 percent

The model performs best on mid-range price levels and struggles more on extreme price categories. This behavior is primarily due to class imbalance and limited feature expressiveness.

## Visualizations

The project includes:

- An interactive [**Folium map**](map_50_by_20_train.html) showing restaurant locations
- [Histograms](price_level_prediction_random_forest_50_by_20.png) comparing **true price levels**, **predicted price levels**, and **prediction error**


These visualizations are used for qualitative model evaluation.

## Example Prediction

```python
new_location = [[32.71, -117.17, 4.2, 350]]
predicted_price_level = model.predict(new_location)
print(predicted_price_level[0])
```

## Limitations

- Price level labels are noisy and frequently missing
- Geographic location alone is a weak predictor
- Class imbalance reduces performance on rare price levels

## Future Improvements

- Add neighborhood-level context such as restaurant density
- Improve encoding of restaurant categories

## Tech Stack

- Python
- Google Places API
- Pandas
- NumPy
- scikit-learn
- Folium
- Matplotlib

## Disclaimer

This project is for educational and portfolio purposes only and is **not affiliated with or endorsed by Google**.
