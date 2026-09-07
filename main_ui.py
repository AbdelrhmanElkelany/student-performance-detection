import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Custom CSS
# ============================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .prediction-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# Helper Functions
# ============================
@st.cache_data
def load_and_process_data(file_path, file_type):
    """تحميل ومعالجة البيانات"""
    df = None
    
    # قراءة الملف حسب النوع
    if file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file_path)
    elif file_type == 'csv':
        # محاولة قراءة CSV بطرق مختلفة
        # أولاً: جرب كـ Excel (بعض الملفات CSV هي Excel فعلياً)
        try:
            df = pd.read_excel(file_path)
        except:
            # ثانياً: جرب كـ CSV بـ encodings وseparators مختلفة
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1256', 'utf-16', 'cp850']
            separators = [',', ';', '\t', '|', ' ']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, engine='python')
                        if len(df.columns) > 1:  # تأكد إن الـ separator صحيح
                            break
                    except:
                        continue
                if df is not None and len(df.columns) > 1:
                    break
        
        if df is None:
            raise ValueError("Unable to read CSV file. Please check the file format and encoding.")
            
    elif file_type == 'txt':
        # محاولة قراءة TXT بطرق مختلفة
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1256', 'utf-16']
        separators = ['\t', ',', ';', '|']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=encoding, engine='python')
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            if df is not None and len(df.columns) > 1:
                break
                
        if df is None:
            raise ValueError("Unable to read TXT file. Please check the file format and encoding.")
            
    elif file_type == 'json':
        try:
            df = pd.read_json(file_path)
        except:
            # محاولة قراءة JSON بطريقة مختلفة
            try:
                df = pd.read_json(file_path, lines=True)
            except:
                raise ValueError("Unable to read JSON file. Please check the file format.")
    
    # Handle Missing Values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include='object').columns
    
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Clean categorical text
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df, df_encoded, categorical_cols

def get_grade_label(score):
    """تحويل الدرجة الرقمية إلى حرف"""
    if score >= 16: return 'A'
    elif score >= 14: return 'B'
    elif score >= 12: return 'C'
    elif score >= 10: return 'D'
    else: return 'F'

@st.cache_resource
def train_models(df_encoded, X):
    """تدريب النماذج"""
    # Target variables
    y_grades = df_encoded['G3'].apply(get_grade_label)
    le = LabelEncoder()
    y_clf = le.fit_transform(y_grades)
    y_reg = df_encoded["G3"]
    
    X_train, X_test, y_train_reg, y_test_reg, y_train_c, y_test_c = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Regression Models
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
    
    # Classification Models
    class_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        ),
        "XGBoost": XGBClassifier(
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
    
    # Train and evaluate regression models
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
    
    # CV for regression
    reg_cv_scores = {}
    for name, model in reg_models.items():
        scores = cross_val_score(model, X_train, y_train_reg, cv=5, scoring='r2')
        reg_cv_scores[name] = scores.mean()
    
    best_reg_name = max(reg_cv_scores, key=reg_cv_scores.get)
    best_reg_model = reg_models[best_reg_name]
    best_reg_model.fit(X_train, y_train_reg)
    
    reg_importances = pd.Series(
        best_reg_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    top_reg_features = reg_importances.head(15).index.tolist()
    
    # Train and evaluate classification models
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
    
    # CV for classification
    cv_scores = {}
    for name, model in class_models.items():
        scores = cross_val_score(model, X_train, y_train_c, cv=5, scoring="f1_weighted")
        cv_scores[name] = scores.mean()
    
    best_model_name = max(cv_scores, key=cv_scores.get)
    best_clf_model = class_models[best_model_name]
    best_clf_model.fit(X_train, y_train_c)
    
    importances = pd.Series(
        best_clf_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    top_features = importances.head(15).index.tolist()
    
    # Final models with top features
    X_train_reg_top = X_train[top_reg_features]
    X_test_reg_top = X_test[top_reg_features]
    final_reg_model = best_reg_model
    final_reg_model.fit(X_train_reg_top, y_train_reg)
    
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    final_clf_model = best_clf_model
    final_clf_model.fit(X_train_top, y_train_c)
    
    return {
        'reg_results': reg_results,
        'class_results': class_results,
        'best_reg_model': final_reg_model,
        'best_clf_model': final_clf_model,
        'top_reg_features': top_reg_features,
        'top_features': top_features,
        'reg_importances': reg_importances,
        'clf_importances': importances,
        'X_test': X_test,
        'y_test_reg': y_test_reg,
        'y_test_c': y_test_c,
        'reg_models': reg_models,
        'class_models': class_models,
        'label_encoder': le,
        'X_train': X_train
    }

def clean_input(data, df, categorical_features, numerical_cols_original):
    """تنظيف المدخلات"""
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

def predict_student(student_dict, models_data, df, categorical_features, final_training_columns, X_train):
    """التنبؤ بأداء الطالب"""
    numerical_cols_original = df.select_dtypes(include=['int64', 'float64']).drop(columns=["G3"], errors='ignore').columns.tolist()
    
    X_new = pd.DataFrame([student_dict])
    X_new = clean_input(X_new, df, categorical_features, numerical_cols_original)
    X_new_encoded = pd.get_dummies(X_new)
    X_new_aligned = X_new_encoded.reindex(columns=final_training_columns)
    
    # Fill missing values
    for col in X_new_aligned.columns:
        if X_new_aligned[col].isnull().any():
            if col in X_train.columns:
                X_new_aligned[col] = X_new_aligned[col].fillna(X_train[col].median())
            else:
                X_new_aligned[col] = X_new_aligned[col].fillna(0)
    
    # Classification
    X_clf = X_new_aligned[models_data['top_features']]
    class_id = models_data['best_clf_model'].predict(X_clf)[0]
    predicted_letter = models_data['label_encoder'].inverse_transform([class_id])[0]
    
    # Regression
    X_reg = X_new_aligned[models_data['top_reg_features']]
    grade = models_data['best_reg_model'].predict(X_reg)[0]
    grade = round(max(0, min(20, grade)), 2)
    
    return grade, predicted_letter

# ============================
# Main App
# ============================
def main():
    st.markdown('<div class="main-header">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Main Menu")
    page = st.sidebar.radio("Choose a page:", 
                           ["📁 Data Overview", 
                            "🤖 Model Performance", 
                            "🔮 Make Predictions",
                            "📈 Visualizations"])
    
    # File uploader
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset", 
        type=['xlsx', 'xls', 'csv', 'txt', 'json'],
        help="Supported formats: Excel (.xlsx, .xls), CSV (.csv), Text (.txt), JSON (.json)"
    )
    
    if uploaded_file is not None:
        # الحصول على نوع الملف
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Load and process data
        df, df_encoded, categorical_cols = load_and_process_data(uploaded_file, file_extension)
        
        # Prepare data for models
        y_reg = df_encoded["G3"]
        X = df_encoded.drop(["G3"], axis=1)
        final_training_columns = X.columns.tolist()
        
        # Train models
        models_data = train_models(df_encoded, X)
        
        # Get feature lists
        top_5_reg = models_data['reg_importances'].head(5).index.tolist()
        top_5_clf = models_data['clf_importances'].head(5).index.tolist()
        selected_features = list(set(top_5_reg + top_5_clf))
        
        numerical_cols_original = df.select_dtypes(include=['int64', 'float64']).drop(columns=["G3"], errors='ignore').columns.tolist()
        numerical_features = [f for f in selected_features if f in numerical_cols_original]
        
        categorical_features = []
        for f in selected_features:
            for original in categorical_cols:
                if f.startswith(original) and original not in categorical_features:
                    categorical_features.append(original)
        
        # ============================
        # Page 1: Data Overview
        # ============================
        if page == "📁 Data Overview":
            st.markdown('<div class="sub-header">Dataset Information</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Pass Rate", f"{(df['G3'] >= 10).mean()*100:.1f}%")
            with col4:
                st.metric("Average Grade", f"{df['G3'].mean():.2f}")
            
            st.markdown("---")
            
            # Grade distribution
            st.markdown("### 📊 Grade Distribution")
            grade_dist = df['G3'].apply(get_grade_label).value_counts().sort_index()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                grade_dist.plot(kind='bar', ax=ax, color=['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71'])
                ax.set_xlabel("Grade Letter")
                ax.set_ylabel("Number of Students")
                ax.set_title("Distribution of Letter Grades")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Grade Breakdown")
                for grade, count in grade_dist.items():
                    percentage = (count / len(df)) * 100
                    st.metric(f"Grade {grade}", f"{count} ({percentage:.1f}%)")
            
            # Display raw data
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.head(20), use_container_width=True)
            
            # Display summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Numerical Summary**")
                st.dataframe(df.describe(), use_container_width=True)
            
            with col2:
                st.markdown("**Categorical Summary**")
                st.dataframe(df.describe(include='object'), use_container_width=True)
            
            # Correlation with G3
            st.markdown('<div class="sub-header">Top Correlations with Final Grade (G3)</div>', unsafe_allow_html=True)
            corr = df_encoded.corr()
            target_corr = corr["G3"].sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=target_corr.values, y=target_corr.index, palette="coolwarm", ax=ax)
            ax.set_xlabel("Correlation with G3")
            ax.set_title("Top 10 Features Correlated with Final Grade")
            st.pyplot(fig)
        
        # ============================
        # Page 2: Model Performance
        # ============================
        elif page == "🤖 Model Performance":
            st.markdown('<div class="sub-header">Model Performance Metrics</div>', unsafe_allow_html=True)
            
            # Regression Results
            st.markdown("### 📊 Regression Models (Predicting Exact Grade)")
            reg_df = pd.DataFrame(models_data['reg_results'])
            
            # تحديد أفضل موديل بناءً على R2 Score
            best_reg_idx = reg_df['R2 Score'].idxmax()
            
            def highlight_regression(s):
                # للأعمدة الرقمية
                if s.name in ['MAE', 'MSE']:
                    # الأقل هو الأفضل - نعمل highlight للأقل قيمة
                    is_min = s == s.min()
                    return ['background-color: #90EE90' if v else '' for v in is_min]
                elif s.name == 'R2 Score':
                    # الأعلى هو الأفضل - نعمل highlight للأعلى قيمة
                    is_max = s == s.max()
                    return ['background-color: #90EE90' if v else '' for v in is_max]
                elif s.name == 'Model':
                    # نعمل highlight على اسم الموديل الأفضل
                    return ['background-color: #FFD700; font-weight: bold' if i == best_reg_idx else '' for i in range(len(s))]
                return [''] * len(s)
            
            st.dataframe(
                reg_df.style.apply(highlight_regression)
                .format({'MAE': '{:.6f}', 'MSE': '{:.6f}', 'R2 Score': '{:.6f}'}),
                use_container_width=True
            )
            
            st.info(f"🏆 Best Regression Model: **{reg_df.loc[best_reg_idx, 'Model']}** (R2 Score: {reg_df.loc[best_reg_idx, 'R2 Score']:.6f})")
            
            # Classification Results
            st.markdown("### 🎯 Classification Models (Predicting Grade Letter)")
            class_df = pd.DataFrame(models_data['class_results'])
            
            # تحديد أفضل موديل بناءً على F1 Score
            best_class_idx = class_df['F1 Score'].idxmax()
            
            def highlight_classification(s):
                # للأعمدة الرقمية - كلها الأعلى هو الأفضل
                if s.name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                    is_max = s == s.max()
                    return ['background-color: #90EE90' if v else '' for v in is_max]
                elif s.name == 'Model':
                    # نعمل highlight على اسم الموديل الأفضل
                    return ['background-color: #FFD700; font-weight: bold' if i == best_class_idx else '' for i in range(len(s))]
                return [''] * len(s)
            
            st.dataframe(
                class_df.style.apply(highlight_classification)
                .format({'Accuracy': '{:.6f}', 'Precision': '{:.6f}', 'Recall': '{:.6f}', 'F1 Score': '{:.6f}'}),
                use_container_width=True
            )
            
            st.info(f"🏆 Best Classification Model: **{class_df.loc[best_class_idx, 'Model']}** (F1 Score: {class_df.loc[best_class_idx, 'F1 Score']:.6f})")
            
            # Grade mapping info
            st.markdown("### 📋 Grade Letter Mapping")
            grade_info = pd.DataFrame({
                'Letter Grade': ['A', 'B', 'C', 'D', 'F'],
                'Score Range': ['16-20', '14-15', '12-13', '10-11', '0-9'],
                'Status': ['Excellent', 'Very Good', 'Good', 'Pass', 'Fail']
            })
            st.dataframe(grade_info, use_container_width=True)
            
            # Feature Importance
            st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Regression Model Features**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(
                    x=models_data['reg_importances'].head(15).values,
                    y=models_data['reg_importances'].head(15).index,
                    palette="viridis",
                    ax=ax
                )
                ax.set_xlabel("Importance Score")
                ax.set_title("Top 15 Regression Features")
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Classification Model Features**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(
                    x=models_data['clf_importances'].head(15).values,
                    y=models_data['clf_importances'].head(15).index,
                    palette="magma",
                    ax=ax
                )
                ax.set_xlabel("Importance Score")
                ax.set_title("Top 15 Classification Features")
                st.pyplot(fig)
        
        # ============================
        # Page 3: Make Predictions
        # ============================
        elif page == "🔮 Make Predictions":
            st.markdown('<div class="sub-header">Student Performance Prediction</div>', unsafe_allow_html=True)
            
            st.info("📝 Enter student information below to predict their final grade and letter grade")
            
            # Create input form
            with st.form("prediction_form"):
                st.markdown("#### Student Information")
                
                # Create columns for better layout
                cols = st.columns(3)
                student_data = {}
                
                # Numerical inputs
                col_idx = 0
                for feature in numerical_features:
                    with cols[col_idx % 3]:
                        if feature in ['G1', 'G2']:
                            student_data[feature] = st.number_input(
                                f"{feature} (0-20)", 
                                min_value=0.0, 
                                max_value=20.0, 
                                value=10.0,
                                step=0.5
                            )
                        elif feature == 'age':
                            student_data[feature] = st.number_input(
                                "Age", 
                                min_value=15, 
                                max_value=22, 
                                value=17
                            )
                        elif feature == 'absences':
                            student_data[feature] = st.number_input(
                                "Absences", 
                                min_value=0, 
                                max_value=100, 
                                value=0
                            )
                        elif feature in ['famrel', 'goout', 'Walc', 'Dalc', 'health']:
                            student_data[feature] = st.slider(
                                f"{feature} (1-5)", 
                                min_value=1, 
                                max_value=5, 
                                value=3
                            )
                        elif feature in ['traveltime', 'studytime']:
                            student_data[feature] = st.slider(
                                f"{feature} (1-4)", 
                                min_value=1, 
                                max_value=4, 
                                value=2
                            )
                        elif feature in ['Medu', 'Fedu']:
                            student_data[feature] = st.slider(
                                f"{feature} (0-4)", 
                                min_value=0, 
                                max_value=4, 
                                value=2
                            )
                        elif feature == 'failures':
                            student_data[feature] = st.slider(
                                "Failures (0-4)", 
                                min_value=0, 
                                max_value=4, 
                                value=0
                            )
                        else:
                            student_data[feature] = st.number_input(
                                feature, 
                                value=0.0
                            )
                    col_idx += 1
                
                # Categorical inputs
                st.markdown("---")
                for feature in categorical_features:
                    with cols[col_idx % 3]:
                        # Check if feature exists in original dataframe
                        if feature in df.columns:
                            options = df[feature].unique().tolist()
                            student_data[feature] = st.selectbox(
                                feature,
                                options=options
                            )
                        else:
                            # البحث عن الـ feature في الأعمدة المشتقة (encoded)
                            related_cols = [col for col in df.columns if col in feature or feature in col]
                            if related_cols:
                                base_feature = related_cols[0]
                                options = df[base_feature].unique().tolist()
                                student_data[base_feature] = st.selectbox(
                                    base_feature,
                                    options=options
                                )
                    col_idx += 1
                
                # Submit button
                st.markdown("---")
                submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)
            
            # معالجة النتائج خارج الـ form
            if submitted:
                try:
                    grade, letter = predict_student(
                        student_data, 
                        models_data, 
                        df, 
                        categorical_features, 
                        final_training_columns,
                        models_data['X_train']
                    )
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Grade", f"{grade}/20")
                    with col2:
                        st.metric("Letter Grade", letter)
                    with col3:
                        if letter in ['A', 'B', 'C', 'D']:
                            st.markdown('<div class="prediction-success">✅ Status: PASS</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="prediction-fail">❌ Status: FAIL</div>', unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(grade/20)
                    
                    # Grade interpretation
                    grade_meanings = {
                        'A': '🌟 Excellent Performance!',
                        'B': '👏 Very Good Performance!',
                        'C': '👍 Good Performance!',
                        'D': '✔️ Passed!',
                        'F': '❌ Failed - Needs Improvement'
                    }
                    st.info(grade_meanings.get(letter, ''))
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please make sure all required fields are filled correctly.")
            
            # Batch prediction
            st.markdown("---")
            st.markdown("### 📊 Batch Prediction")
            st.info("Upload a CSV/Excel file with student data for batch predictions")
            
            batch_file = st.file_uploader("Upload file for batch prediction", type=['csv', 'xlsx', 'xls'])
            if batch_file is not None:
                batch_df = None
                file_ext = batch_file.name.split('.')[-1].lower()
                
                # محاولة قراءة Excel أولاً
                if file_ext in ['xlsx', 'xls']:
                    try:
                        batch_df = pd.read_excel(batch_file)
                    except Exception as e:
                        st.error(f"Error reading Excel file: {str(e)}")
                else:
                    # محاولة قراءة كـ Excel حتى لو الامتداد CSV
                    try:
                        batch_df = pd.read_excel(batch_file)
                    except:
                        # محاولة قراءة كـ CSV بـ encodings وseparators مختلفة
                        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1256', 'utf-16', 'cp850']
                        separators = [',', ';', '\t', '|', ' ']
                        
                        for encoding in encodings:
                            for sep in separators:
                                try:
                                    # إعادة تعيين pointer الملف
                                    batch_file.seek(0)
                                    batch_df = pd.read_csv(batch_file, encoding=encoding, sep=sep, engine='python')
                                    if len(batch_df.columns) > 1:  # تأكد إن الـ separator صحيح
                                        break
                                except:
                                    continue
                            if batch_df is not None and len(batch_df.columns) > 1:
                                break
                
                if batch_df is not None and len(batch_df) > 0:
                    st.success(f"✅ File loaded successfully! Found {len(batch_df)} students with {len(batch_df.columns)} columns.")
                    
                    # عرض preview من البيانات
                    with st.expander("📋 Preview Data"):
                        st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("▶️ Start Batch Prediction", use_container_width=True):
                        predictions = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in batch_df.iterrows():
                            try:
                                student_dict = row.to_dict()
                                grade, letter = predict_student(
                                    student_dict, 
                                    models_data, 
                                    df, 
                                    categorical_features, 
                                    final_training_columns,
                                    models_data['X_train']
                                )
                                predictions.append({
                                    'Student': idx+1, 
                                    'Predicted Grade': grade, 
                                    'Letter Grade': letter,
                                    'Status': 'Pass' if letter in ['A', 'B', 'C', 'D'] else 'Fail'
                                })
                                
                                # Update progress
                                progress = (idx + 1) / len(batch_df)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {idx+1}/{len(batch_df)} students")
                            except Exception as e:
                                predictions.append({
                                    'Student': idx+1, 
                                    'Predicted Grade': 'Error', 
                                    'Letter Grade': 'Error',
                                    'Status': f'Error: {str(e)}'
                                })
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Batch prediction completed!")
                        
                        results_df = pd.DataFrame(predictions)
                        
                        # عرض إحصائيات
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Students", len(results_df))
                        with col2:
                            passed = len(results_df[results_df['Status'] == 'Pass'])
                            st.metric("Passed", passed)
                        with col3:
                            failed = len(results_df[results_df['Status'] == 'Fail'])
                            st.metric("Failed", failed)
                        with col4:
                            pass_rate = (passed / len(results_df)) * 100 if len(results_df) > 0 else 0
                            st.metric("Pass Rate", f"{pass_rate:.1f}%")
                        
                        # Grade distribution
                        st.markdown("#### 📊 Grade Distribution")
                        grade_counts = results_df['Letter Grade'].value_counts()
                        fig, ax = plt.subplots(figsize=(10, 5))
                        grade_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c'])
                        ax.set_xlabel("Letter Grade")
                        ax.set_ylabel("Number of Students")
                        ax.set_title("Distribution of Predicted Grades")
                        plt.xticks(rotation=0)
                        st.pyplot(fig)
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            "📥 Download Predictions",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                else:
                    st.error("❌ Unable to read the file. Please check:")
                    st.markdown("""
                    - File format is correct (CSV or Excel)
                    - File is not corrupted
                    - File contains data
                    - Column separators are standard (comma, semicolon, tab)
                    
                    **Tip:** Try saving your file as Excel (.xlsx) format for better compatibility.
                    """)
        
        # ============================
        # Page 4: Visualizations
        # ============================
        elif page == "📈 Visualizations":
            st.markdown('<div class="sub-header">Data Visualizations</div>', unsafe_allow_html=True)
            
            # Grade distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Numerical Grade Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['G3'], bins=20, kde=True, ax=ax, color='#3498db')
                ax.set_xlabel("Final Grade (G3)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Final Grades")
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 🎯 Letter Grade Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                grade_counts = df['G3'].apply(get_grade_label).value_counts().sort_index()
                colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
                ax.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title("Letter Grade Distribution")
                st.pyplot(fig)
            
            # Pass/Fail analysis
            st.markdown("### ✅ Pass/Fail Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                pass_fail = (df['G3'] >= 10).map({True: 'Pass', False: 'Fail'}).value_counts()
                colors = ['#90ee90', '#ff9999']
                ax.pie(pass_fail.values, labels=pass_fail.index, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title("Pass vs Fail Distribution")
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Statistics")
                total = len(df)
                passed = (df['G3'] >= 10).sum()
                failed = total - passed
                st.metric("Total Students", total)
                st.metric("Passed (≥10)", f"{passed} ({(passed/total*100):.1f}%)")
                st.metric("Failed (<10)", f"{failed} ({(failed/total*100):.1f}%)")
            
            # Grades by feature
            st.markdown("### 📈 Grades by Feature")
            categorical_options = df.select_dtypes(include='object').columns.tolist()
            if categorical_options:
                feature_to_plot = st.selectbox("Select feature:", categorical_options)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, x=feature_to_plot, y='G3', ax=ax, palette='Set2')
                    ax.set_title(f"Grade Distribution by {feature_to_plot}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(data=df, x=feature_to_plot, y='G3', ax=ax, palette='muted')
                    ax.set_title(f"Grade Density by {feature_to_plot}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Correlation heatmap
            st.markdown("### 🔥 Correlation Heatmap (Top Features)")
            top_features_for_heatmap = models_data['top_features'][:10] + ['G3']
            # Filter only existing columns
            top_features_for_heatmap = [f for f in top_features_for_heatmap if f in df_encoded.columns]
            corr_subset = df_encoded[top_features_for_heatmap].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, 
                       square=True, linewidths=1)
            ax.set_title("Correlation Heatmap - Top Features")
            st.pyplot(fig)
            
            # Confusion Matrix for best classification model
            st.markdown("### 🎯 Confusion Matrix - Best Classification Model")
            best_clf = models_data['best_clf_model']
            y_pred = best_clf.predict(models_data['X_test'][models_data['top_features']])
            cm = confusion_matrix(models_data['y_test_c'], y_pred)
            
            class_labels = models_data['label_encoder'].classes_
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_labels, 
                       yticklabels=class_labels, ax=ax)
            ax.set_xlabel("Predicted Grade")
            ax.set_ylabel("Actual Grade")
            ax.set_title("Confusion Matrix - Grade Classification")
            st.pyplot(fig)
            
            # Scatter plots for top features
            st.markdown("### 🔍 Feature vs Grade Analysis")
            numerical_features_available = [f for f in models_data['top_reg_features'][:5] 
                                           if f in df_encoded.columns and f != 'G3']
            
            if len(numerical_features_available) >= 2:
                col1, col2 = st.columns(2)
                
                for idx, feature in enumerate(numerical_features_available[:4]):
                    with col1 if idx % 2 == 0 else col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.scatterplot(data=df_encoded, x=feature, y='G3', alpha=0.6, ax=ax)
                        ax.set_title(f"{feature} vs Final Grade")
                        st.pyplot(fig)
    
    else:
        st.info("👈 Please upload your dataset from the sidebar to begin")
        st.markdown("""
        ### 📋 Instructions:
        1. Upload your student performance dataset from the sidebar
        2. Navigate through different pages using the sidebar
        3. Explore data, model performance, and make predictions
        
        ### 📁 Supported File Formats:
        - **Excel**: .xlsx, .xls
        - **CSV**: .csv
        - **Text**: .txt (tab or comma separated)
        - **JSON**: .json
        
        ### 📊 Features:
        - **Data Overview**: View dataset statistics and correlations
        - **Model Performance**: Compare different ML models
        - **Make Predictions**: Predict student performance with letter grades (A, B, C, D, F)
        - **Visualizations**: Interactive charts and graphs
        
        ### 🎓 Grade System:
        - **A (16-20)**: Excellent
        - **B (14-15)**: Very Good
        - **C (12-13)**: Good
        - **D (10-11)**: Pass
        - **F (0-9)**: Fail
        """)

if __name__ == "__main__":
    main()