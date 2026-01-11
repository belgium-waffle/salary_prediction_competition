# FINAL: Enhanced pipeline (CatBoost + LightGBM + robust FE + Ridge meta-blend)
import os, gc, time, warnings
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")
BASE_PATH = "/kaggle/input/eee-g513"
SEED = 42
np.random.seed(SEED)

# ----------------------------
# 0) Helper functions
# ----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def add_oof_target_encoding(train_df, test_df, col, target_col="log_salary",
                            n_splits=5, group_col="city", noise=0.0, seed=SEED):
    """
    Out-of-fold target encoding with optional gaussian noise on validation folds (reduces overfitting).
    Uses GroupKFold over group_col to avoid leakage when needed.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = pd.Series(index=train_df.index, dtype=float)
    rnd = np.random.RandomState(seed)
    for tr_idx, val_idx in gkf.split(train_df, groups=train_df[group_col]):
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]
        fold_mean = tr.groupby(col)[target_col].mean()
        val_te = val[col].map(fold_mean)
        if noise and val_te.notna().any():
            # add small gaussian noise only to validation mapped values
            val_te = val_te + rnd.normal(0, noise, size=len(val_te))
        oof.iloc[val_idx] = val_te
    # Fill remaining with global mean
    global_mean = train_df[target_col].mean()
    oof = oof.fillna(global_mean)
    train_df[f"{col}_te"] = oof
    # For test: map using full-train group mean (no noise)
    full_mean = train_df.groupby(col)[target_col].mean()
    test_df[f"{col}_te"] = test_df[col].map(full_mean).fillna(global_mean)
    return train_df, test_df

# ----------------------------
# 1) Load data
# ----------------------------
train = pd.read_csv(f"{BASE_PATH}/train.csv")
test  = pd.read_csv(f"{BASE_PATH}/test.csv")
cost  = pd.read_csv(f"{BASE_PATH}/cost_of_living.csv", na_values=["NA","Na","na",""])

print("Loaded:", train.shape, test.shape, cost.shape)

# ----------------------------
# 2) Prepare cost columns (numeric + medians) BEFORE merge
# ----------------------------
col_features = [c for c in cost.columns if c.startswith("col_")]
for c in col_features:
    cost[c] = pd.to_numeric(cost[c], errors="coerce")
    cost[c] = cost[c].fillna(cost[c].median())

# ----------------------------
# 3) Merge cost info into train/test
# ----------------------------
merge_keys = ["country", "state", "city"]
train = train.merge(cost, on=merge_keys, how="left")
test  = test.merge(cost, on=merge_keys, how="left")
print("After merge:", train.shape, test.shape)

# ----------------------------
# 4) Target cleaning & log transform
# ----------------------------
train = train[train["salary_average"].notna()]
train = train[train["salary_average"] > 0]
train = train.reset_index(drop=True)
train["log_salary"] = np.log1p(train["salary_average"])
print("Target cleaned. NaNs in target:", train["log_salary"].isna().sum())

# ----------------------------
# 5) Fill any remaining numeric NaNs from merge with column medians
# ----------------------------
for c in col_features:
    if c in train.columns:
        med = cost[c].median() if c in cost.columns else 0.0
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)

# ----------------------------
# 6) Feature engineering: PCA + KMeans on cost features
# ----------------------------
all_cost = pd.concat([train[col_features], test[col_features]], axis=0).reset_index(drop=True)
scaler = StandardScaler()
all_cost_s = scaler.fit_transform(all_cost)

pca = PCA(n_components=10, random_state=SEED)
all_pca = pca.fit_transform(all_cost_s)
pca_cols = [f"pca_{i}" for i in range(all_pca.shape[1])]
train[pca_cols] = all_pca[:len(train)]
test[pca_cols]  = all_pca[len(train):]

kmeans = KMeans(n_clusters=8, random_state=SEED, n_init=10)
all_cluster = kmeans.fit_predict(all_cost_s)
train["cost_cluster"] = all_cluster[:len(train)]
test["cost_cluster"]  = all_cluster[len(train):]

# ----------------------------
# 7) Aggregation features: use MEDIAN for stability
# ----------------------------
agg_cols = col_features.copy()
country_agg = train.groupby("country")[agg_cols].median().add_suffix("_country_med")
state_agg   = train.groupby("state")[agg_cols].median().add_suffix("_state_med")

train = train.merge(country_agg, left_on="country", right_index=True, how="left")
test  = test.merge(country_agg, left_on="country", right_index=True, how="left")
train = train.merge(state_agg, left_on="state", right_index=True, how="left")
test  = test.merge(state_agg, left_on="state", right_index=True, how="left")

# Fill new agg NaNs with column medians
for c in list(country_agg.columns) + list(state_agg.columns):
    if c in train.columns:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)

# ----------------------------
# 8) Frequency encoding base categoricals
# ----------------------------
cat_base = ["country", "state", "city", "role"]
for c in cat_base:
    train[f"{c}_freq"] = train[c].map(train[c].value_counts())
    test[f"{c}_freq"]  = test[c].map(train[c].value_counts()).fillna(0).astype(int)

# ----------------------------
# 9) OOF target encoding (role, country, city) with a touch of noise
# ----------------------------
# noise small (0.01) to regularize TE
train, test = add_oof_target_encoding(train, test, "role",    target_col="log_salary", n_splits=5, group_col="city",    noise=0.01, seed=SEED)
train, test = add_oof_target_encoding(train, test, "country", target_col="log_salary", n_splits=5, group_col="city",    noise=0.01, seed=SEED)
train, test = add_oof_target_encoding(train, test, "city",    target_col="log_salary", n_splits=5, group_col="country", noise=0.01, seed=SEED)
print("Added OOF TE: role_te, country_te, city_te")

# ----------------------------
# 10) Interaction feature: city_role frequency (low-risk)
# ----------------------------
train["city_role"] = train["city"].astype(str) + "_" + train["role"].astype(str)
test["city_role"]  = test["city"].astype(str) + "_" + test["role"].astype(str)
freq_cr = train["city_role"].value_counts()
train["city_role_freq"] = train["city_role"].map(freq_cr)
test["city_role_freq"]  = test["city_role"].map(freq_cr).fillna(0).astype(int)

# ----------------------------
# 11) Label Encoding for categorical columns (for LGBM)
# ----------------------------
cat_features = ["country", "state", "city", "role", "country_role", "cost_cluster"]
# ensure country_role exists
train["country_role"] = train["country"].astype(str) + "_" + train["role"].astype(str)
test["country_role"]  = test["country"].astype(str) + "_" + test["role"].astype(str)

label_encoders = {}
for c in cat_features:
    # if column missing (country_role added above), ensure present
    if c not in train.columns:
        train[c] = train[c].astype(str)
        test[c]  = test[c].astype(str)
    le = LabelEncoder()
    le.fit(list(train[c].astype(str).values) + list(test[c].astype(str).values))
    train[c + "_le"] = le.transform(train[c].astype(str))
    test[c + "_le"]  = le.transform(test[c].astype(str))
    label_encoders[c] = le

# ----------------------------
# 12) Final feature list construction
# ----------------------------
feature_candidates = []
feature_candidates += col_features
feature_candidates += pca_cols
feature_candidates += ["cost_cluster"]
feature_candidates += [f"{c}_freq" for c in cat_base]
feature_candidates += ["role_te", "country_te", "city_te"]
feature_candidates += ["city_role_freq"]
# add a manageable slice of country_agg columns for signal
feature_candidates += list(country_agg.columns[:10])
# label-encoded categorical columns for LGBM
feature_candidates += [c + "_le" for c in cat_features]

# Keep only existing columns
features = [f for f in feature_candidates if f in train.columns]
print("Total features used:", len(features))

# Fill remaining NaNs
train[features] = train[features].fillna(0)
test[features]  = test[features].fillna(0)

# ----------------------------
# 13) CV setup
# ----------------------------
groups = train["city"]
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
y = train["log_salary"].values

# ----------------------------
# 14) Train CatBoost (OOF)
# ----------------------------
oof_cat = np.zeros(len(train))
preds_cat = np.zeros(len(test))

cat_params = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 9,
    "l2_leaf_reg": 6.0,        # stronger regularization
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "early_stopping_rounds": 150,
    "verbose": 200,
    "thread_count": -1
}

cat_models = []
start = time.time()
for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, groups=groups)):
    print(f"\nCatBoost Fold {fold+1}")
    X_tr = train.iloc[tr_idx][features]
    X_val = train.iloc[val_idx][features]
    y_tr = y[tr_idx]; y_val = y[val_idx]
    # pass label-encoded cat features (names exist in features list)
    cat_feat_names = [c + "_le" for c in cat_features if (c + "_le") in features]
    train_pool = Pool(X_tr, y_tr, cat_features=cat_feat_names)
    val_pool   = Pool(X_val, y_val, cat_features=cat_feat_names)
    model = CatBoostRegressor(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cat[val_idx] = model.predict(X_val)
    preds_cat += model.predict(test[features]) / n_splits
    cat_models.append(model)
    fold_rmse = rmse(y_val, oof_cat[val_idx])
    print(f"CatBoost Fold {fold+1} RMSE (log): {fold_rmse:.5f}")
print("CatBoost training done in", round(time.time()-start,1), "s")

# ----------------------------
# 15) Train LightGBM (OOF)
# ----------------------------
oof_lgb = np.zeros(len(train))
preds_lgb = np.zeros(len(test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,   # helps regularization
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "verbose": -1,
    "seed": SEED
}

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, groups=groups)):
    print(f"\nLightGBM Fold {fold+1}")
    X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = train.iloc[tr_idx]["log_salary"], train.iloc[val_idx]["log_salary"]

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200)
        ]
    )

    oof_lgb[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    preds_lgb += model.predict(test[features], num_iteration=model.best_iteration) / n_splits

    fold_rmse = rmse(y_val, oof_lgb[val_idx])
    print(f"LightGBM Fold {fold+1} RMSE (log): {fold_rmse:.5f}")

# quick OOF diagnostics
print("\nOOF CatBoost (log):", rmse(train["log_salary"], oof_cat))
print("OOF LightGBM (log):", rmse(train["log_salary"], oof_lgb))

# ----------------------------
# 16) Ridge meta-blend: alpha CV search + stability
# ----------------------------
# Prepare meta inputs
X_meta_train = np.vstack([oof_cat, oof_lgb]).T
X_meta_test  = np.vstack([preds_cat, preds_lgb]).T
y_meta = train["log_salary"].values

# standardize
scaler_meta = StandardScaler()
X_meta_train_s = scaler_meta.fit_transform(X_meta_train)
X_meta_test_s  = scaler_meta.transform(X_meta_test)

# small search over alphas (fast)
alphas = [0.1, 0.3, 1.0, 3.0]
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
best_alpha, best_cv = None, 1e9
print("\nSearching best Ridge alpha via 5-fold CV...")
for a in alphas:
    cvs = []
    for tr_i, val_i in kf.split(X_meta_train_s):
        r = Ridge(alpha=a)
        r.fit(X_meta_train_s[tr_i], y_meta[tr_i])
        p = r.predict(X_meta_train_s[val_i])
        cvs.append(rmse(y_meta[val_i], p))
    mean_cv = np.mean(cvs)
    print(f"alpha={a} → CV log-RMSE={mean_cv:.5f}")
    if mean_cv < best_cv:
        best_cv, best_alpha = mean_cv, a

print(f"Selected alpha={best_alpha} (CV log-RMSE={best_cv:.5f})")

final_ridge = Ridge(alpha=best_alpha, random_state=SEED)
final_ridge.fit(X_meta_train_s, y_meta)
oof_blend = final_ridge.predict(X_meta_train_s)
submission_preds = final_ridge.predict(X_meta_test_s)

oof_rmse_log = rmse(train["log_salary"], oof_blend)
oof_rmse_real = rmse(train["salary_average"], np.expm1(oof_blend))
print(f"\n✅ Final Ridge meta-blend RMSE (log):  {oof_rmse_log:.5f}")
print(f"✅ Final Ridge meta-blend RMSE (real): {oof_rmse_real:.2f}")
print("Ridge coefficients (CatBoost, LightGBM):", final_ridge.coef_)

# ----------------------------
# 17) Final submission
# ----------------------------
submission = pd.DataFrame({
    "ID": test["ID"],
    "salary_average": np.expm1(submission_preds)
})
submission["salary_average"] = submission["salary_average"].clip(lower=1000, upper=400000)
submission.to_csv("submission_blend_final_v2.csv", index=False)
print("\n✅ Final blended submission saved as 'submission_blend_final_v2.csv'")

# ----------------------------
# 18) Cleanup
# ----------------------------
gc.collect()
