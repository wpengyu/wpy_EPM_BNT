# 🔋 Predictive Modeling of Energy Storage Properties of BNT-Based Ceramics

**Machine Learning & Experimental Study**

---

### 🧪 Overview

This project aims to predict energy storage performance of BNT-based perovskite ceramics using machine learning techniques. It involves the generation of candidate chemical formulas, feature engineering using elemental descriptors, and property prediction via ensemble regression models.

---

### 🔁 Workflow Summary

1. **Feature Generation**  
   Parse chemical formulas and compute descriptor-based features using elemental properties.

2. **Formula Generation**  
   Use Bayesian optimization (Hyperopt) to randomly generate new A-site and B-site element combinations.

3. **Feature Calculation for Candidates**  
   Compute features for each generated formula.

4. **Prediction**  
   Load pre-trained models (SVR, Gradient Boosting, AdaBoost) and a meta-model to predict:
   - `Target 1`: Energy storage density (Wrec)  
   - `Target 2`: Efficiency (n)

5. **Saving Results**  
   Output the predicted formulas and values into a CSV file.

---

### 📁 File Descriptions

| File | Description |
|------|-------------|
| `content.xlsx` | Elemental content/proportions for each ceramic sample. |
| `descriptor.xlsx` | Elemental physical descriptors (e.g., atomic radius, electronegativity, etc.). |
| `w&n.xlsx` | Experimental measurements: `Wrec` (energy density) and `n` (efficiency). *(Not included in repo)* |
| `generated_formulas_with_features.csv` | Computed descriptors for generated formulas. |
| `generated_formulas_with_predictions.csv` | Predicted `Wrec` and `n` values using machine learning models. |

---

## 🛠️ How It Works — Code Flow

### 1. Feature Generation (Code Block 1)
Parses chemical formulas, separates A-site/B-site elements, and computes 60+ features using `descriptor.xlsx`.

---

### 2. Formula Generation (Code Block 2)
Uses **Hyperopt** to create random ABO₃ candidate formulas through probabilistic search.

---

### 3. Save Features (Code Block 3)
Outputs all generated formulas and descriptors to `generated_formulas_with_features.csv`.

---

### 4. Predict Target Properties (Code Block 4)
Loads descriptor CSV, selects features for two targets, and uses pre-trained machine learning models:

- **SVR** (Support Vector Regressor)  
- **Gradient Boosting Regressor**  
- **AdaBoost Regressor**  

The final prediction is made using a **meta-model** trained on the base model outputs.

---

### 5. Save Predictions (Code Block 5)
Predictions for `Wrec` and `n` are saved in `generated_formulas_with_predictions.csv`.

---

> ⚠️ **PS: About Data Availability**  
> Due to ongoing research, the original lab experimental data (`w&n.xlsx`) is not publicly available.  




