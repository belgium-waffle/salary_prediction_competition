# 🏆 Salary Prediction Competition - 1st Place Solution

![Competition Banner](https://img.shields.io/badge/Rank-1st%20Place-gold?style=for-the-badge&logo=kaggle) ![Language](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python) ![Model](https://img.shields.io/badge/Model-CatBoost%20%2B%20LightGBM%20%2B%20Ridge-orange?style=for-the-badge)

## 📌 Overview
This repository contains the **1st place solution** for the **EEE G513 Machine Learning for EE** Salary Prediction Competition (T2 Evaluative Component). The objective was to predict average salaries for various job roles across 1,000+ cities globally, accounting for complex local economic conditions.

Our winning approach utilizes a robust **Stacking Ensemble** of **CatBoost** and **LightGBM** regressors, blended with a **Ridge Regression** meta-learner. We heavily focused on **feature engineering** to capture the non-linear relationship between 52 cost-of-living indicators and salary distributions.

## 📊 Dataset
The competition dataset combined salary information with comprehensive cost-of-living indicators.

* **`train.csv`**: 6,525 samples (Salary data for specific roles in specific cities).
* **`test.csv`**: 2,799 samples from 311 cities (Predictions required for these city-role combinations).
    * *Note: There is no city overlap between training and test sets, requiring strong generalization.*
* **`cost_of_living.csv`**: 52 economic indicators for 1,528 cities (housing, food, transport, healthcare, etc.).

## 💡 Solution Architecture

### 1. Data Preprocessing & Cleaning
* **Log-Transformation:** Applied `np.log1p` to the target variable (`salary_average`) to normalize the skewed salary distribution.
* **Imputation:** Filled missing cost-of-living data with median values to handle data gaps gracefully.
* **Consistency:** Standardized city/country names for accurate merging across datasets.

### 2. Advanced Feature Engineering
We engineered a rich set of features to capture local economic contexts:
* **PCA & Clustering:** Performed PCA (10 components) and K-Means Clustering (8 clusters) on cost-of-living features to create dense representations of economic status.
* **Aggregations:** Calculated median cost-of-living stats grouped by `country` and `state` to provide regional context to individual cities.
* **Target Encoding (OOF):** Implemented **Out-of-Fold (OOF) Target Encoding** for `role`, `country`, and `city` using GroupKFold. We added Gaussian noise to validation folds to prevent overfitting.
* **Frequency Encoding:** Encoded categorical variables based on their frequency.
* **Interaction Features:** Created combined features like `city_role` to capture specific local demand for roles.

### 3. Model Pipeline
Our final model is a **Stacking Ensemble**:

1.  **Level 1 Models (Base Learners):**
    * **CatBoost Regressor:** Optimized for handling categorical features natively.
        * *Key Params:* `depth=9`, `l2_leaf_reg=6.0`, `iterations=3000`, `learning_rate=0.03`
    * **LightGBM Regressor:** Fast gradient boosting with label encoding.
        * *Key Params:* `num_leaves=128`, `feature_fraction=0.8`, `lambda_l1=0.1`

2.  **Cross-Validation Strategy:**
    * **5-Fold GroupKFold:** We split data based on `city` groups. This ensured that the model was validated on *unseen cities*, accurately simulating the test set conditions.

3.  **Level 2 Model (Meta-Learner):**
    * **Ridge Regression:** Blends the OOF predictions from CatBoost and LightGBM.
    * *Why Ridge?* It handles multicollinearity between the two strong base models effectively.
    * *Alpha Tuning:* Automated CV search selected `alpha=0.1` for optimal blending.

## 📈 Performance
The model achieved top performance on the private leaderboard.

| Metric | Score (RMSPE) |
| :--- | :--- |
| **Private Leaderboard Score** | **0.08990** |
| **Final Ensemble CV (Log RMSE)** | **0.171** |

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/salary-prediction-competition.git](https://github.com/yourusername/salary-prediction-competition.git)
    cd salary-prediction-competition
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn lightgbm catboost
    ```

3.  **Place Data:**
    Ensure `train.csv`, `test.csv`, and `cost_of_living.csv` are in the project root (or update `BASE_PATH` in the script).

4.  **Run the training script:**
    ```bash
    python main.py
    ```
    *This will train the models, run cross-validation, and generate `submission_blend_final_v2.csv`.*

