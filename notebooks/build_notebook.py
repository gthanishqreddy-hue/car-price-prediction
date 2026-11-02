"""Regenerate notebooks/exploratory_analysis.ipynb from the project's own modules.

Usage (from the project root):
    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory_analysis.ipynb
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Car Price Prediction - Exploratory Data Analysis\n\n"
       "This notebook is a guided walk through the data. All logic lives in `src/` so the "
       "notebook and the training pipeline cannot drift apart."),
    code("import sys\nfrom pathlib import Path\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n\n"
         "import pandas as pd\n"
         "from IPython.display import Image, display\n"
         "from src import config\n"
         "from src.data_preprocessing import (load_raw_data, clean_data, is_two_wheeler,\n"
         "                                    train_test_split_stratified, iqr_outlier_summary)\n"
         "from src.eda import build_eda_frame, run_eda\n"
         "pd.set_option('display.width', 120)"),
    md("## 1. Load and inspect the raw data"),
    code("raw = load_raw_data()\nprint(raw.shape)\nraw.head()"),
    code("raw.info()"),
    code("print('Missing values per column:')\nprint(raw.isna().sum())\n"
         "print('\\nExact duplicate rows:', raw.duplicated().sum())"),
    code("raw.describe().T"),
    md("## 2. Data-quality finding: the file mixes cars and two-wheelers\n\n"
       "`Car_Name` reveals motorcycles and scooters (Bajaj, Royal Enfield, Activa, ...). "
       "They form one contiguous block of rows, i.e. a second dataset appended to the cars."),
    code("bikes = is_two_wheeler(raw['Car_Name'])\n"
         "print('two-wheeler rows:', bikes.sum(), '| index range:', raw.index[bikes].min(), "
         "'-', raw.index[bikes].max())\n"
         "raw.loc[bikes, ['Car_Name', 'Present_Price', 'Selling_Price', 'Kms_Driven']].head(8)"),
    code("clean, report = clean_data(raw)\nreport"),
    md("After scoping to cars, the extreme 500,000 km value (an Activa scooter) is gone and "
       "`Seller_Type` becomes very imbalanced - individual sellers were mostly bike listings."),
    code("for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']:\n"
         "    print(clean[col].value_counts().to_dict())"),
    md("## 3. Train/test split first, then explore the training set only"),
    code("X_train, X_test, y_train, y_test = train_test_split_stratified(clean)\n"
         "print(len(X_train), 'train rows /', len(X_test), 'test rows')\n"
         "eda = build_eda_frame(X_train, y_train)\n"
         "pd.DataFrame(iqr_outlier_summary(eda, ['Selling_Price', 'Present_Price', "
         "'Kms_Driven', 'Car_Age']))"),
    md("## 4. Figures\n\n`run_eda()` writes every figure to `artifacts/figures/`."),
    code("summary = run_eda()\nsummary['spearman_with_target_train']"),
    code("for name in summary['figures']:\n"
         "    print(name)\n"
         "    display(Image(filename=str(config.FIGURES_DIR / name)))"),
    md("## 5. Takeaways that shaped the model\n\n"
       "* Price is right-skewed and spans two orders of magnitude -> evaluate log and ratio "
       "targets.\n"
       "* `Present_Price` is the strongest single signal; age is the second.\n"
       "* Retained value (selling / present price) is similar for manual and automatic cars, "
       "so the automatic premium mostly reflects more expensive models.\n"
       "* The single 92.6-lakh car is far outside the rest of the data; tree models cannot "
       "extrapolate to it on an absolute price scale, which motivated the ratio target."),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("exploratory_analysis.ipynb")
nbf.write(nb, out)
print(f"Wrote {out}")
