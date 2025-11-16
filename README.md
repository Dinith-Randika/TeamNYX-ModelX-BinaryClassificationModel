# TeamNYX-ModelX-BinaryClassificationModel

# Dementia Risk Prediction From Non Medical Factors

This repository contains the notebook and report for predicting dementia status using only non medical and simple self reported features from the NACC dataset.
The work was developed for a machine learning competition that evaluates modeling, data handling, and interpretability.

---

## 1. Project Overview

**Goal**

Build a binary classification model that predicts whether a participant is demented or not using non medical features such as demographics, functional status, lifestyle, comorbidities, and family history.

**Main steps in the notebook**

1. Load the NACC dataset from a CSV file.
2. Select non medical features and engineer additional clinically meaningful variables.
3. Preprocess the data and create train and test splits.
4. Train baseline and advanced models.
5. Tune hyperparameters.
6. Evaluate models using several performance metrics.
7. Interpret the final model using feature importance and SHAP.

**Main models**

* Logistic Regression (baseline)
* CatBoostClassifier (final model)

The final tuned CatBoost model achieved strong performance on the held out test set and was chosen as the final model.

---

## 2. Repository Contents

Depending on how you organize the repo, the structure will be similar to:

* `notebooks/` or root folder

  * `Dementia_Prediction_Non_Medical.ipynb`
* `report/`

  * Final PDF or Word report (optional)
* `README.md` (this file)

If you rename the notebook, please adjust the filename in this section.

---

## 3. Environment Setup

### 3.1. Software requirements

* Python 3.9 or later
* Jupyter Notebook or JupyterLab

### 3.2. Python packages

The notebook uses the following main libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `catboost`
* `shap`
* `matplotlib`

You can install them using:

```bash
pip install pandas numpy scikit-learn catboost shap matplotlib
```

If you prefer a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # on Linux or macOS
# or
.venv\Scripts\activate         # on Windows

pip install -r requirements.txt   # if you create one
# or use the pip command above
```

---

## 4. Data

### 4.1. Dataset source

The notebook expects a CSV file named:

```text
Dementia Prediction Dataset.csv
```

The original dataset is not included in this repository because of size and licensing constraints.
Please follow the competition or course instructions to obtain the CSV file.

### 4.2. Important note about the file path

In the notebook, the dataset is loaded using a path that points to the author local machine:

```python
file_path = r"C:\Users\ASUS\Downloads\Dementia Prediction Dataset.csv"
df = pd.read_csv(file_path, nrows=10000)
```

This path will not exist on another computer.

To run the notebook, you only need to change the value of `file_path` to point to the location of `Dementia Prediction Dataset.csv` on your own machine.

#### Recommended simple setup

1. Create a folder named `data` in the root of the repository.
2. Place `Dementia Prediction Dataset.csv` inside the `data` folder.
3. Edit the first data loading cell in the notebook to:

   ```python
   file_path = "data/Dementia Prediction Dataset.csv"
   df = pd.read_csv(file_path, nrows=10000)
   ```

If you wish to use the full dataset, you can remove the `nrows=10000` argument:

```python
df = pd.read_csv(file_path)
```

---

## 5. How To Run The Notebook

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set up the Python environment**

   Install the required packages as described in section 3.

3. **Prepare the dataset**

   * Obtain `Dementia Prediction Dataset.csv` from the competition data source.
   * Place it in a convenient location (for example, `data/Dementia Prediction Dataset.csv`).
   * Open the notebook and update the `file_path` variable in the first data loading cell so that it points to your local copy.

4. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

   Then open `Dementia_Prediction_Non_Medical.ipynb` (or the main notebook file in this repo).

5. **Run the notebook**

   * Run all cells sequentially from top to bottom.
   * The notebook will:

     * Load and explore the data.
     * Perform feature engineering and preprocessing.
     * Split data into train and test sets.
     * Train and tune models.
     * Print evaluation metrics.
     * Generate feature importance and SHAP plots.

Execution time depends on the machine, but it should be reasonable on a modern laptop.

---

## 6. Reproducibility Notes For Judges

If you wish to check that the implementation matches the written report, the key places to look are:

* **Data exploration and feature engineering**
  Early sections where non medical features are selected and composite variables like smoking burden and comorbidity counts are created.

* **Model building and hyperparameter tuning**
  Cells that define the CatBoost and Logistic Regression models, the preprocessing pipelines, and the `RandomizedSearchCV` tuning for CatBoost.

* **Model evaluation**
  Cells that compute ROC AUC, classification report, confusion matrix, and SHAP based explainability plots.

As long as the dataset path is updated correctly, the notebook should run without further changes.

---

## 7. Contact

If there are questions about running the notebook or about any modeling choice, please use the contact channel provided in the competition instructions.
