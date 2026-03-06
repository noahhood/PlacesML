# PlacesML: Restaurant Price Level Prediction with Machine Learning

**PlacesML** predicts a restaurant’s **price level** using location, popularity, and category signals from the **Google Places API**. The project demonstrates end-to-end machine learning from feature engineering to model evaluation. The final model is a **Random Forest classifier** achieving approximately **68% accuracy** on a held-out test set.

---

## Project Goal

Estimate a restaurant’s price level (`0–5`, as defined by Google Places) using widely available features. This is a **multiclass classification** problem, framed to help understand pricing patterns in a city or region.

---

## Data Collection

Restaurant data is collected using the **Google Places Nearby Search API** by scanning geographic grids and querying nearby restaurants. Results are deduplicated by Place ID. Only restaurants with complete metadata are retained.

**Fields used:**
- Latitude & Longitude
- Rating
- User rating count
- Price level (target)
- Primary type
- Types (used to engineer categorical features)

---

## Feature Engineering

Several new features were added to improve prediction:

- **Primary type encoding (`primary_type_enc`)** – Label-encoded category of the restaurant.  
- **Fine dining flag (`fine_dining`)** – `True` if the restaurant is classified as fine dining.  
- **Fast food flag (`fast_food`)** – `True` if the restaurant is classified as fast food.  

These features complement location and popularity metrics, providing better context for price prediction.

**Final input features:**
- Latitude
- Longitude
- Rating
- User rating count
- Primary type encoding
- Fine dining flag
- Fast food flag

**Target Variable:** `price_level` (categorical integer 0–5)

---

## Model

A **RandomForestClassifier** is used with the following configuration:

- Number of trees: 200  
- Class balancing: enabled (`balanced`)  
- Train/test split: 80% / 20%  
- Cross-validation: Stratified 5-Fold  

The model is trained on all engineered features and evaluated using cross-validation to ensure stability and mitigate class imbalance.

---

## Results

**Train/Test split:** 609 / 153 samples  

**Cross-Validation Metrics:**
- F1 weighted: 0.63  
- Accuracy: 0.65  

**Test Set Metrics:**
- F1 weighted: 0.66  
- Accuracy: 0.68  

The model performs best on mid-range price levels (`2–3`) and struggles on rare extremes (`4–5`) due to class imbalance and limited representation in the dataset.

---

## Visualizations

The project includes:

- **Interactive Folium map** showing restaurant locations colored by price level.  
- **Histograms** comparing true price levels, predicted price levels, and prediction errors.  

These visualizations provide insight into geographic patterns and model performance.

---

## Example Usage

The model can predict a restaurant’s price level from location, rating, user count, and category indicators.

---

## Limitations

- Price level labels are noisy and sometimes missing  
- Geographic location and popularity metrics alone are weak predictors  
- Rare price levels (`4–5`) are difficult to predict due to class imbalance  

---

## Future Improvements

- Incorporate neighborhood-level context such as restaurant density or median income  
- Include more nuanced category information (e.g., cuisine type, chain vs. independent)  
- Explore advanced machine learning models such as XGBoost or neural networks  
- Apply oversampling or class-balancing techniques for rare price levels  

---

## Tech Stack

- Python  
- Google Places API  
- Pandas & NumPy  
- scikit-learn  
- Matplotlib & Folium  

---

## Disclaimer

This project is for educational and portfolio purposes only and is **not affiliated with or endorsed by Google**.
