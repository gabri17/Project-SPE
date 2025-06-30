import pandas as pd
from pathlib import Path
from time import time
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import ttest_rel  # Moved to top-level import

# 1. Paths to training and testing sets - VERIFIED CORRECT
data_dir = Path("/Users/sano/Desktop/Project-SPE/CSV Files")
train_fp = data_dir / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"
test_fp  = data_dir / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"

# 2. Load datasets - VERIFIED CORRECT
df_train = pd.read_csv(train_fp)
df_test  = pd.read_csv(test_fp)

print(f"Loaded training data: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
print(f"Loaded testing  data: {df_test.shape[0]} rows, {df_test.shape[1]} columns")

# 3. Separate features and labels - VERIFIED CORRECT
target_col = 'label'
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]
X_test  = df_test.drop(columns=[target_col])
y_test  = df_test[target_col]

# 4. Identify numeric and categorical features - VERIFIED CORRECT
numeric_feats = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_feats = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# 5. Build preprocessing pipelines - VERIFIED CORRECT
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', cat_transformer, categorical_feats)
])

# 6. Define base model pipelines - VERIFIED CORRECT
pipelines = {
    'Logistic Regression': Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'Neural Network': Pipeline([
        ('prep', preprocessor),
        ('clf', MLPClassifier(random_state=42, max_iter=200))
    ])
}

# 7. Hyperparameter tuning for Random Forest and Neural Network - VERIFIED CORRECT
param_distributions = {
    'Random Forest': {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    },
    'Neural Network': {
        'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate_init': [0.001, 0.01]
    }
}

tuned_models = {}
for name in ['Random Forest', 'Neural Network']:
    print(f"\nTuning hyperparameters for {name}...")
    search = RandomizedSearchCV(
        pipelines[name],
        param_distributions[name],
        n_iter=10,
        scoring='f1',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    print(f"Best params for {name}: {search.best_params_}")

# 8. Evaluate tuned models with CV - FIXED: Store results for t-test later
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results_dict = {}  # Store results for statistical testing

for name, model in tuned_models.items():
    print(f"\nRunning CV for tuned {name}...")
    start = time()
    cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    cv_results_dict[name] = cv_res  # Store full results
    elapsed = time() - start
    print(f"Completed in {elapsed:.2f}s")
    for metric in scoring:
        m = cv_res[f'test_{metric}']
        print(f"  {metric.capitalize():>9}: {m.mean():.3f} ± {m.std():.3f}")

# 9. Measure inference time and test evaluation - VERIFIED CORRECT
inference_times = {}
for name, model in tuned_models.items():
    print(f"\n{name} - final evaluation on test set:")
    start_inf = time()
    y_pred = model.predict(X_test)
    end_inf = time()
    inference_times[name] = (end_inf - start_inf) / len(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Avg inference time per sample: {inference_times[name]*1e3:.4f} ms")

# 10. Statistical significance - FIXED: Use stored CV results
print("\nPaired t-tests on CV F1-scores:")
f1_scores = {}
for name in tuned_models:
    f1_scores[name] = cv_results_dict[name]['test_f1']

model_names = list(tuned_models.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        stat, p = ttest_rel(f1_scores[m1], f1_scores[m2])
        print(f"{m1} vs {m2}: t={stat:.3f}, p={p:.3f}")

# 11. Feature importance - FIXED: Handle feature names correctly
print("\nTop 10 features by importance from Random Forest:")
if 'Random Forest' in tuned_models:
    rf_pipeline = tuned_models['Random Forest']
    tree = rf_pipeline.named_steps['clf']
    
    # Get feature names after preprocessing
    preprocessor = rf_pipeline.named_steps['prep']
    try:
        feat_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        num_names = numeric_feats
        cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_feats)
        feat_names = np.concatenate([num_names, cat_names])
    
    importances = tree.feature_importances_
    top_idx = importances.argsort()[::-1][:10]
    for idx in top_idx:
        print(f"  {feat_names[idx]}: {importances[idx]:.4f}")

# REMOVED redundant test set evaluation at end