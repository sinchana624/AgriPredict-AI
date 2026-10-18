# 🌾 AgriPredict AI

### AI-Powered Crop Yield Prediction, Pesticide Usage Analysis and Agricultural Decision Support System

AgriPredict AI is a machine-learning-based agricultural analytics project designed to explore historical crop-yield trends and generate experimental crop-yield estimates using historical crop and weather information.

The application provides an interactive dashboard where users can explore agricultural data, generate predictions, review model evaluation metrics, manage their profiles, and download PDF reports.

## ✨ Features

- **User Authentication:** Local sign-up, login, and logout using SQLite.
- **Profile Management:** Update your name and user role.
- **Overview Dashboard:** View dataset summaries and agricultural insights.
- **Crop Yield Prediction:** Generate experimental yield estimates using year, crop, and weather inputs.
- **Explore Historical Data:** Filter crop-yield records and visualize historical trends.
- **Model Performance:** Review test MAE, RMSE, and R² metrics.
- **PDF Report Generation:** Generate and download reports containing available project information and prediction results.
- **Interactive Visualizations:** Explore agricultural data through interactive charts.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly
- **Model Storage:** Joblib
- **Local Authentication Database:** SQLite
- **PDF Reports:** ReportLab

## 📁 Project Structure

```text
AgriPredict-AI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── processed/
│   │   └── combined_crop_yield.csv
│   └── agripredict_users.db
├── models/
│   └── yield_model.joblib
├── reports/
│   └── model_metrics.csv
├── src/
│   ├── predict.py
│   └── train_model.py
├── tests/
└── .gitignore
```

The SQLite database is created locally when users register. It should not be uploaded to the public repository.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sinchana624/AgriPredict-AI.git
cd AgriPredict-AI
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If ReportLab is not included in `requirements.txt`, install it separately:

```bash
python -m pip install reportlab
```

### 4. Verify required files

Ensure that the project contains the processed dataset, trained model, and model metrics:

- `data/processed/combined_crop_yield.csv`
- `models/yield_model.joblib`
- `reports/model_metrics.csv`

If these files are not present, follow the model-training instructions in the project before launching the dashboard.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## 🔐 Authentication

AgriPredict AI uses a local SQLite database for account registration and authentication.

Users can:

1. Create an account with their name, email, and password.
2. Sign in using their registered credentials.
3. Update their profile information.
4. Sign out of the application.

The user database is stored locally and is not shared automatically across different computers.

**Security note:** This authentication setup is intended for a local project demonstration. A public deployment requires additional security measures and an appropriate production authentication solution.

## 📊 Model Evaluation

The dashboard displays the available model evaluation metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted yields.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
- **R² (Coefficient of Determination):** Describes how well the model explains variation in the evaluation data.

The metrics are loaded from `reports/model_metrics.csv`. Their values depend on the model-training pipeline and dataset.

## ⚠️ Limitations

- Predictions are experimental estimates, not guaranteed crop yields.
- Dataset coverage and quality can affect model performance.
- Historical trends may not represent future agricultural conditions.
- Weather inputs alone do not capture soil quality, irrigation, pests, farming practices, or other important factors.
- The model requires further independent validation before being used for real agricultural decisions.
- Local SQLite accounts are not automatically available across devices.

The application should be treated as an educational and experimental decision-support prototype, not as a substitute for professional agricultural advice.

## 🚀 Future Enhancements

- Integrate live and historical weather APIs.
- Improve model validation