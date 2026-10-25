# Heart Disease Prediction Using Machine Learning

This is a complete major-project package built from the supplied project brief and heart dataset. It includes a written report, a reproducible model-training script, analysis figures and result tables, a saved model, and an optional Streamlit prediction demo.

## Project contents

- `Heart_Disease_Prediction_Report.pdf` - formatted project report.
- `REPORT.md` - editable report text.
- `train_models.py` - cleaning, exploratory analysis, model tuning, evaluation, and artifact generation.
- `app.py` - optional Streamlit demo using the saved model.
- `data/heart.csv` - the supplied worksheet exported as a real CSV.
- `data/heart_source_original.xlsx` - original workbook bytes, renamed to match its actual format.
- `artifacts/` - trained model, comparison results, summary metadata, and charts.
- `requirements.txt` - Python dependencies.

## Run the analysis

Use Python 3.10 or newer. From this folder:

```bash
python -m pip install -r requirements.txt
python train_models.py
python build_report.py
```

The script saves or refreshes the model and figures in `artifacts/`. The supplied file contains 303 records, no missing cells, and one exact duplicate row. The script removes that duplicate before the stratified split, leaving 302 rows for analysis. Imputers are still included in each model pipeline so the workflow can handle missing inputs if the dataset changes.

The saved model was created with scikit-learn 1.7.2, which is pinned in `requirements.txt` for compatibility.

## Launch the optional prediction demo

After installing the requirements and training the model:

```bash
streamlit run app.py
```

The demo uses the categories as numeric codes because the supplied file does not contain a verified codebook for every category. It only accepts values in the observed dataset ranges.

## Method summary

- Target: `target` (0 or 1), as supplied.
- Inputs: 13 feature columns.
- Numeric measurements: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`; median imputation and standard scaling are applied in the pipeline.
- Coded categories: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`; most-frequent imputation and one-hot encoding are applied in the pipeline.
- Evaluation: stratified 80/20 holdout split with `random_state=42`; 5-fold stratified cross-validation is used for grid search on the training portion only.
- Models: Logistic Regression, Decision Tree, Random Forest, and a small feedforward neural network (`MLPClassifier`).
- Selection: the final model is chosen by the highest mean cross-validation ROC-AUC. The test set is reserved for the reported comparison.

## Important scope note

This is an educational classification exercise using a small historical dataset. Its output is not a diagnosis, clinical risk estimate, or substitute for professional care. Performance on this sample does not establish safety or usefulness in a real clinical setting.

## Data format note

The file supplied as `heart.csv` is actually an Excel workbook (ZIP-based `.xlsx` data) with a worksheet named `in`. The original bytes are included as `data/heart_source_original.xlsx`; the worksheet was exported without changing its values to the proper CSV at `data/heart.csv`.

## References

- User-supplied project brief: *Heart Disease Prediction Using Machine Learning Classification Models*.
- UCI Machine Learning Repository, [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease). The supplied file's exact provenance and recoding history are not independently verified.
- scikit-learn documentation: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
