import pandas as pd

# ============================
# 1. Load Dataset
# ============================
df = pd.read_excel("Portuguese.csv")

print("\n===== RAW DATA LOADED =====")
print(df.head())

# ============================
# 2. Basic info
# ============================
print("\n===== INFO =====")
df.info()

# ============================
# 3. Column names
# ============================
print("\n===== FEATURES =====")
print(list(df.columns))

# ============================
# 4. Numerical Summary
# ============================
print("\n===== NUMERIC SUMMARY =====")
print(df.describe())

# ============================
# 5. Categorical Summary
# ============================
print("\n===== CATEGORICAL SUMMARY =====")
print(df.describe(include='object'))

# ============================
# 6. Unique values for categorical columns
# ============================
print("\n===== UNIQUE VALUES =====")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].unique())


# ============================
# 7. Handle Missing Values
# ============================
print("\n===== HANDLING MISSING VALUES =====")

# Numerical → fill median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical → fill mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ============================
# 8. Clean categorical text (lowercase & strip)
# ============================
print("\n===== CLEANING CATEGORICAL VALUES =====")
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# ============================
# 9. One-hot encode categorical columns
# ============================
print("\n===== ENCODING CATEGORICAL FEATURES =====")
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\n===== ENCODED FEATURES =====")
print(df_encoded.head())

# ============================
# 10. Export final processed dataset
# ============================
df_encoded.to_csv("processed_student_data.csv", index=False)
print("\n===== EXPORT COMPLETE: processed_student_data.csv =====")

# ============================
# 11. CORRELATION MATRIX (FOCUSED ON G3)
# ============================

import seaborn as sns
import matplotlib.pyplot as plt

# Full correlation matrix
corr = df_encoded.corr()

# Export correlation with G3 only
target_corr = corr["G3"].sort_values(ascending=False)
target_corr.to_csv("G3_correlations.csv")

print("\n===== CORRELATION WITH G3 =====")
print(target_corr)

# Optional: heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

print("\n===== CORRELATION EXPORT COMPLETE =====")

# ============================
# 12. TRAIN/TEST SPLIT
# ============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Target variable
def get_grade_label(score):
    if score >= 16: return 'A'
    elif score >= 14: return 'B'
    elif score >= 12: return 'C'
    elif score >= 10: return 'D'
    else: return 'F'

y_grades = df_encoded['G3'].apply(get_grade_label)

le = LabelEncoder()
y_clf = le.fit_transform(y_grades)

print("\n===== Class Mapping =====")
for i, label in enumerate(le.classes_):
    print(f"Class {i} represents: {label}")
    
y_reg = df_encoded["G3"]

# Features (everything except G3)
X = df_encoded.drop(["G3"], axis=1)

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train_reg, y_test_reg, y_train_c, y_test_c = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ============================
# 13. Regression Models (Baseline)
# ============================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

reg_models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    ),
    "Linear Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
}

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

reg_results = []
for name, model in reg_models.items():
    model.fit(X_train, y_train_reg)
    y_pred = model.predict(X_test)
    
    reg_results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test_reg, y_pred),
        "MSE": mean_squared_error(y_test_reg, y_pred),
        "R2 Score": r2_score(y_test_reg, y_pred)
    })
    
print("\n==== Regression Results ====")
print(pd.DataFrame(reg_results))

# ============================
# 14. Regression Cross-Validation & Feature Importance
# ============================
from sklearn.model_selection import cross_val_score
reg_cv_scores = {}

for name, model in reg_models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train_reg,
        cv=5,
        scoring='r2'
    )
    reg_cv_scores[name] = scores.mean()
    
print("\n==== Regression CV Scores (R2) ====")
for k, v in reg_cv_scores.items():
    print(k, ":", round(v, 4))
    
best_reg_name = max(reg_cv_scores, key=reg_cv_scores.get)
best_reg_model = reg_models[best_reg_name]

print("\nBest Regression Model:", best_reg_name)

best_reg_model.fit(X_train, y_train_reg)

reg_importances = pd.Series(
    best_reg_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

top_reg_features = reg_importances.head(15).index.tolist()

print("\n==== Top 15 Regression Features ====")
print(top_reg_features)

plt.figure(figsize=(10, 8))
sns.barplot(
    x=reg_importances.head(15).values,
    y=reg_importances.head(15).index,
    palette="viridis"
)
plt.title("Top 15 Regression Feature Importances (Predicting G3)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("reg_feature_importance.png")
plt.show()

# ============================  
# 15. Regression Models Comparison Plots  
# ============================
plt.figure(figsize=(18, 5))

for i, (name, model) in enumerate(reg_models.items(), 1):
    y_pred = model.predict(X_test)
    
    plt.subplot(1, 3, i)
    sns.scatterplot(x=y_test_reg, y=y_pred, alpha=0.6)
    
    min_val = min(y_test_reg.min(), y_pred.min())
    max_val = max(y_test_reg.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(f"{name}")
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")

plt.tight_layout()
plt.show()

# ============================
# 16. Final Regression Model (Top Features)
# ============================
X_train_reg_top = X_train[top_reg_features]
X_test_reg_top = X_test[top_reg_features]

final_reg_model = best_reg_model

final_reg_model.fit(X_train_reg_top, y_train_reg)
y_pred_reg_top = final_reg_model.predict(X_test_reg_top)

print("\n==== Final Regression Performance (Top Features) ====")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg_top))
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg_top))
print("R2:", r2_score(y_test_reg, y_pred_reg_top))

# ============================
# 17. Classification Models
# ============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    ),
    "XGBoost":
        XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42        
    ),
    "SVM (RBF Kernel)": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])
}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class_results = []
for name, model in class_models.items():
    model.fit(X_train, y_train_c)
    y_pred_c = model.predict(X_test)
    
    class_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_c, y_pred_c),
        "Precision": precision_score(y_test_c, y_pred_c, average='weighted', zero_division=0),
        "Recall": recall_score(y_test_c, y_pred_c, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test_c, y_pred_c, average='weighted', zero_division=0)
    })
    
print("\n==== Classification Results ====")
print(pd.DataFrame(class_results))

# ============================
# 18. Classification Cross-Validation & Feature Importance
# ============================
cv_scores = {}
for name, model in class_models.items():
    scores = cross_val_score(
        model, X_train, y_train_c, cv=5, scoring="f1_weighted"
        )
    
    cv_scores[name] = scores.mean()
    
best_model_name = max(cv_scores, key=cv_scores.get)
best_clf_model = class_models[best_model_name]
    
print("\nBest Model Based on CV F1:", best_clf_model)

best_clf_model.fit(X_train, y_train_c)

importances = pd.Series(
    best_clf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

top_features = importances.head(15).index.tolist()

print("\n==== Top 15 Classification Features ====")
print(top_features)

plt.figure(figsize=(10, 8))
sns.barplot(
    x=importances.head(15).values,
    y=importances.head(15).index,
    hue=importances.head(15).index,
    legend=False,
    palette="magma"
)
plt.title("Top 15 Classification Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("clf_feature_importance.png")
plt.show()

# ============================
# 19. Classification Models Comparison Plots
# ============================
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(18, 5))
class_labels = le.classes_

for i, (name, model) in enumerate(class_models.items(), 1):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test_c, y_pred)
    
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Grade")
    plt.ylabel("Actual Grade")

plt.tight_layout()
plt.show()

# ============================
# 20. Final Classification Model (Top Features)
# ============================
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

final_clf_model = best_clf_model
final_clf_model.fit(X_train_top, y_train_c)

# ============================
# 21. Input Cleaning Function
# ============================
final_training_columns = X.columns.tolist()
categorical_cols_original = df.select_dtypes(include='object').columns.tolist()
numerical_cols_original = df.select_dtypes(include=['int64', 'float64']).drop(columns=["G3"]).columns.tolist()

top_5_reg = reg_importances.head(5).index.tolist()
top_5_clf = importances.head(5).index.tolist()

selected_features = list(set(top_5_reg + top_5_clf))

numerical_features = [f for f in selected_features if f in numerical_cols_original]
categorical_features = []

for f in selected_features:
    for original in categorical_cols_original:
        if f.startswith(original) and original not in categorical_features:
            categorical_features.append(original)

print("\n==== Feature Selection Complete ====")
print(f"Total features to be requested: {len(selected_features)}")
print(f"Features: {selected_features}")

import numpy as np

def clean_input(data):
    rules_numeric = {
        'G1': (0, 20), 'G2': (0, 20), 'absences': (0, 100),
        'age': (15, 22), 'famrel': (1, 5), 'goout': (1, 5),
        'Walc': (1, 5), 'Dalc': (1, 5),
        'traveltime': (1, 4), 'Medu': (0, 4), 'Fedu': (0, 4),
        'studytime': (1, 4), 'failures': (0, 4), 'health': (1, 5)
    }

    for col in data.columns:
        if col in rules_numeric:
            low, high = rules_numeric[col]
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(df[col].median())
            data[col] = np.clip(data[col], low, high)
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.lower()
    return data

# ============================
# 22. Predict Function
# ============================

def predict_student(student_dict):
    X_new = pd.DataFrame([student_dict])
    X_new = clean_input(X_new)
    
    X_new_encoded = pd.get_dummies(X_new)
    X_new_aligned = X_new_encoded.reindex(columns=final_training_columns)
    
    for col in X_new_aligned.columns:
        if X_new_aligned[col].isnull().any():
            if col in X_train.columns:
                X_new_aligned[col] = X_new_aligned[col].fillna(X_train[col].median())
            else:
                X_new_aligned[col] = X_new_aligned[col].fillna(0)

    # Classification
    X_clf_input = X_new_aligned[top_features]
    class_id = final_clf_model.predict(X_clf_input)[0]
    predicted_letter = le.inverse_transform([class_id])[0]
    
    # Regression
    X_reg_input = X_new_aligned[top_reg_features]
    predicted_score = final_reg_model.predict(X_reg_input)[0]
    predicted_score = round(max(0, min(20, predicted_score)), 2)

    return predicted_score, predicted_letter

# ============================
# 23. Unified Interactive Input Function
# ============================
def input_student_data():
    students = []
    print("\nEnter students' data (press Q to exit at any time):")

    categorical_options = {col: df[col].unique().tolist() for col in categorical_features}

    while True:
        student = {}
        print("\n--- A New Student ---")
        
        # Numeric inputs
        for feature in numerical_features:
            while True:
                val = input(f"{feature} (numeric): ")
                if val.lower() == 'q':
                    return students
                try:
                    student[feature] = float(val)
                    break
                except ValueError:
                    print("Please enter a valid number.")

        # Categorical inputs
        for feature in categorical_features:
            options = ', '.join(map(str, categorical_options[feature]))
            while True:
                val = input(f"{feature} (options: {options}): ").strip()
                if val.lower() == 'q':
                    return students
                if val:
                    student[feature] = val
                    break

        students.append(student)
        print("Student added! Enter next student or press Q to finish.\n")

# ===== Run interactive input =====
students_input = input_student_data()
for i, student in enumerate(students_input, 1):
    grade, status = predict_student(student)
    print(f"\nStudent {i}:")
    print("Predicted Grade:", grade)
    print("Predicted Status:", status)