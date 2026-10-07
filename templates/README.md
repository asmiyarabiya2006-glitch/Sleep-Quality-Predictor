# Sleep Quality Predictor

A machine learning web app that predicts sleep quality (Good / Average / Poor) from daily habits and gives personalized tips.

## Features
- Predicts sleep quality from sleep duration, exercise and stress level
- Personalized tips for better sleep
- Past predictions table and trend chart
- Calm blue and purple design

## Technologies
- Python, Flask
- scikit-learn, pandas, NumPy
- Matplotlib, Seaborn
- Chart.js

## Dataset
Sleep Health and Lifestyle Dataset (Kaggle)

## Models Compared
Logistic Regression, Decision Tree, Random Forest, SVM.
Random Forest was used in the app (about 97% test accuracy).

## How to Run
1. Install libraries: `pip install pandas numpy scikit-learn matplotlib seaborn flask joblib`
2. Put `sleep.csv` inside the `data` folder
3. Train the model: `python train.py`
4. Start the app: `python app.py`
5. Open http://127.0.0.1:5000 in your browser