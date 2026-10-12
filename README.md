# ⚖️ Impact of Imbalanced Data on Model Accuracy

A data mining project showing how **class imbalance** misleads a machine learning model, and how **SMOTE-Tomek** balancing fixes it, using employee attrition prediction as the case study.

## How we did it

1. **Dataset:** IBM HR Analytics Employee Attrition: 1,470 employees, 35 attributes. Only 16% left, an imbalance of **5.2 : 1**.
2. **Preprocessing:** label encoding, `RobustScaler` scaling, and a stratified 80/20 train/test split.
3. **Model 1:** a Random Forest trained on the imbalanced data as-is.
4. **Balancing:** SMOTE-Tomek on the training set (synthetic "left" examples plus removal of noisy borderline cases), bringing it to 1 : 1.
5. **Model 2:** the same Random Forest trained on the balanced data, with a decision threshold tuned for the best F1 score.
6. **Comparison:** metrics, confusion matrices and probability distributions on the same test set.

## Results (test set: 294 employees, 47 left)

| Metric (for "Left") | Model 1: Imbalanced | Model 2: Balanced |
|---------------------|---------------------|-------------------|
| Accuracy            | 83.3%               | 81.3%             |
| Precision           | 0.40                | 0.45              |
| **Recall**          | **0.09**            | **0.77**          |
| **F1 score**        | **0.14**            | **0.57**          |

**Key finding:** Model 1 looks accurate, but it caught only **4 of 47** employees who left. Model 2 caught **36 of 47** for just a 2% drop in accuracy. **Accuracy alone is misleading on imbalanced data.**

## What's in each folder

- **`DataMiningProject_Code.ipynb`**: the full analysis, with code, explanations and charts
- **`dataset/`**: the HR attrition dataset (CSV)
- **`documentation/`**: project report and presentation (PDF)

## View the results

All outputs and charts are saved in the notebook, so **no need to run anything**: click the notebook here on GitHub, or download the project and open it in VS Code, Jupyter or Google Colab, and the results show right away.

## Run it yourself (optional)

```bash
git clone https://github.com/violazeneli31/ImbalancedDataSmoteAnalysis.git
cd ImbalancedDataSmoteAnalysis

# Check your version with python3 --version; if you have several, use the newest (e.g. python3.12)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook DataMiningProject_Code.ipynb
```

Or open the notebook in **VS Code** (with the Python and Jupyter extensions), select the `.venv` as the kernel and click **Run All**.

> Re-running replaces the saved outputs, and results may differ slightly depending on library versions, because SMOTE-Tomek balancing involves randomness.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)

## Authors

Sidrit Rrustemi, Viola Zeneli and Artemisa Hasalami
