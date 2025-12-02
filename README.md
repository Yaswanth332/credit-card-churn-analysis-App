<b>🔍 Credit Card Customer Churn Prediction (ML + Streamlit)</b>

This project predicts whether a bank customer is likely to churn using machine learning.
It includes full end-to-end development:

✔ Data Cleaning
✔ Feature Engineering
✔ SMOTE Balancing
✔ Model Training (RandomForest & XGBoost)
✔ Hyperparameter Tuning
✔ ROC-based Threshold Optimization
✔ Deployment using Streamlit

🚀 Tech Stack
Component	Technology
Programming	Python
ML Framework	Scikit-Learn, XGBoost
Data	Pandas, NumPy
Deployment	Streamlit
Model Persistence	Joblib

📈 Model Performance (Final Selected Model)
Metric	Score
Accuracy	0.9536
Precision	0.8123
Recall	0.9266
F1 Score	0.8657

Model selected using the business objective of maximizing recall.

🧠 How to Run

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app/app.py

📁 Project Structure
📦 credit-card-churn-analysis-App
 ┣ 📂 data
 ┣ 📂 model
 ┣ 📂 app
 ┣ 📂 notebooks
 ┣ README.md
 ┣ requirements.txt

📜 License

MIT License (optional)
