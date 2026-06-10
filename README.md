# CUSTOMER-CHURN-PREDICTION

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Machine Learning project focused on predicting customer churn using classification models and customer behavior data. The project analyzes the `Churn_Modelling.csv` dataset, compares multiple algorithms, and exposes a simple Flask API for prediction.

## Project Overview

This project identifies customers who are likely to leave a business or subscription service. It uses exploratory analysis, preprocessing, model training, and evaluation to support better customer retention decisions.

## Key Highlights

- Built and compared Logistic Regression, Random Forest, and Gradient Boosting models.
- Applied preprocessing with scaling and one-hot encoding for mixed feature types.
- Tuned the Random Forest model using `GridSearchCV`.
- Saved the best trained model for reuse in an API workflow.
- Added a Flask inference API for single-customer churn prediction.

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask

## Repository Structure

```text
.
|-- CUSTOMER CHURN.ipynb
|-- app.py
|-- data/
|   `-- Churn_Modelling.csv
|-- examples/
|   `-- sample_request.json
|-- models/
|-- src/
|   `-- train.py
|-- requirements.txt
`-- LICENSE
```

## Dataset

- Source file included in this repository: [data/Churn_Modelling.csv](data/Churn_Modelling.csv)
- Original reference note: [Link for DataSet](./Link%20for%20DataSet)

## Model Performance

Based on the notebook results:

- Logistic Regression accuracy: `0.81`
- Random Forest accuracy: `0.86`
- Gradient Boosting accuracy: `0.87`
- Tuned Random Forest accuracy: `0.87`
- Best tuned Random Forest parameters: `n_estimators=200`, `max_depth=None`, `min_samples_split=10`

## How To Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/train.py
```

This command creates `models/churn_model.pkl`, which is ignored from Git tracking.

4. Start the API:

```bash
python app.py
```

## Sample Prediction Request

Use the example file at [examples/sample_request.json](examples/sample_request.json), or send a request like this:

```json
{
  "features": {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
  }
}
```

## License

This project is available under the MIT License.

## Author

Nadeem Ahamad

Machine Learning project focused on Customer Churn Prediction, customer risk classification, churn pattern analysis, and model evaluation using Python by CodSoft Internship Project.
