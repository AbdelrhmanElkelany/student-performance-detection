# 🎓 Student Performance Prediction System

## 🔍 Key Features & Highlights

### 📊 Advanced Data Visualization
- Numeric and letter grade distributions
- Pass / Fail analysis
- Correlation heatmaps with final grades (G3)
- Feature importance analysis for both regression and classification models
- Interactive dashboards for clear insights

### 🧠 Complete Machine Learning Pipeline
- Robust data preprocessing (handling missing values, encoding, scaling)
- Multiple regression models for predicting the final numeric grade (G3):
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
- Multiple classification models for predicting the letter grade (A–F):
  - Random Forest Classifier
  - XGBoost Classifier
  - SVM (RBF Kernel)
- Model comparison and automatic selection of the best-performing model via cross-validation

### 📈 Proper Model Evaluation
- Accuracy, Precision, Recall, and F1-Score (classification)
- R² Score, MAE, and MSE (regression)
- 5-fold cross-validation to ensure model reliability

### 🧪 Batch Prediction Capability
- Predicting performance for multiple students simultaneously
- Uploading datasets (Excel `.xlsx`/`.xls`, CSV, TXT, JSON formats)
- Generating summary statistics and insights
- Exporting prediction results for further analysis

### 🌐 Interactive Web Application (Streamlit)
The app (`main_ui.py`) is organized into 4 pages, navigable from the sidebar:
- **📁 Data Overview** — explore the uploaded dataset
- **🤖 Model Performance** — compare regression & classification models
- **🔮 Make Predictions** — individual and batch prediction workflows
- **📈 Visualizations** — grade distributions, correlations, feature importance

Designed for both technical and non-technical users, with real-time visual feedback.

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── Maths.csv              # Math course student records (397 rows, 33 features)
├── Portuguese.csv         # Portuguese course student records (651 rows, 33 features)
├── src_code.py            # Standalone data science script: EDA, preprocessing,
│                          # correlation analysis, model training & evaluation
├── main_ui.py             # Streamlit web application (interactive UI)
├── ML_Presantation.pptx   # Project presentation slides
└── README.md
```

> ⚠️ **Technical note:** `Maths.csv` and `Portuguese.csv` are saved with a `.csv`
> extension but are actually stored in Excel (`.xlsx`) binary format. This is
> intentional/handled in the code — both `src_code.py` and `main_ui.py` read
> them with `pd.read_excel(...)` (with a CSV fallback in the Streamlit app),
> so keep the `.csv` names as-is unless you also update the loading logic.

### Dataset columns (both files share the same schema)
`school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3`

Source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/USERNAME/student-performance-prediction.git
cd student-performance-prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost seaborn matplotlib streamlit
```

> If you add a `requirements.txt` to the repo, replace the line above with:
> `pip install -r requirements.txt`

### Running the Streamlit App

```bash
streamlit run main_ui.py
```

Then open your browser at `http://localhost:8501`, and upload `Maths.csv` or
`Portuguese.csv` (or your own dataset with a compatible schema) from the sidebar.

### Running the Standalone Analysis Script

```bash
python src_code.py
```

This reproduces the full offline pipeline: EDA → cleaning → encoding →
correlation analysis → model training/evaluation → interactive console-based
predictions. It also exports:
- `processed_student_data.csv`
- `G3_correlations.csv`
- `correlation_heatmap.png`
- `reg_feature_importance.png`
- `clf_feature_importance.png`

---

## 🔧 Ongoing Development & Future Improvements

This project is actively being improved and extended, with a focus on:

- Enhancing prediction accuracy through better feature engineering
- Experimenting with additional models and hyperparameter tuning
- Improving data quality handling and generalization
- Expanding visual analytics and dashboard capabilities
- Moving closer to a production-ready educational tool

---

## 🎯 Why This Project Matters

This project is:
- **Not** just a notebook
- **Not** just a university assignment
- **But** a scalable foundation for a real educational decision-support system that can be used, expanded, and integrated into real-world environments.

---

## 📊 Presentation

See `ML_Presantation.pptx` for the full project walkthrough and results summary.
