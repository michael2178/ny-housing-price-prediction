# NY Housing Price Prediction

Predicting New York City property listing prices with linear regression, and comparing it against Random Forest and Gradient Boosting.

The full analysis is in [`NY_Housing_Price_Prediction.ipynb`](NY_Housing_Price_Prediction.ipynb): data cleaning, exploratory analysis, outlier handling, multicollinearity checks, model fitting, interpretation and evaluation.

## Key results

| Model | Features | R² (log price) | MAE (USD) |
|---|---|---|---|
| Baseline linear regression | BEDS, BATH, PROPERTYSQFT, LATITUDE, LONGITUDE | 0.521 | 634,464 |
| Expanded linear regression | Baseline + borough + property type | **0.748** | **458,616** |
| Random Forest | Same as expanded | 0.843 | 340,404 |
| Gradient Boosting | Same as expanded | 0.845 | 345,008 |

All models are evaluated on the same held-out test set (557 listings, 20% of the cleaned data). R² is measured on log price and MAE in dollars.

## Key findings

- **Data quality mattered.** Of the 4,801 listings, 214 were exact duplicates, 1,709 had mean-imputed (placeholder) values for square footage or bathrooms, and 96 were implausible records (for example a price of about \$2.15 billion). The final dataset has 2,782 listings.
- **Log-transforming price worked.** Raw prices are heavily right-skewed (mean \$1.31M vs median \$0.80M after filtering); log price is close to symmetric.
- **Multicollinearity was not severe.** `BEDS`, `BATH` and `PROPERTYSQFT` are strongly correlated (about 0.7 to 0.8), but all VIF values are below 4.
- **Location and property type matter.** Adding borough and property type raised R² from 0.52 to 0.75 and cut the average dollar error by about 28%. Median prices range from about \$590K in the Bronx to about \$1.74M in Manhattan.
- **Coefficients (expanded model, all else equal):** a Condo is about 79% pricier than a Co-op and a House about 71%; Manhattan is about 172% pricier than the Bronx; each extra bathroom is about +11% and each extra 100 sq ft about +5%.
- **Non-linear models do better.** Random Forest and Gradient Boosting reach R² of about 0.84 to 0.85. Square footage is the most important feature in the Random Forest. Linear regression is kept as the main model because its coefficients are easy to interpret.

## Methodology

1. Load the data, remove duplicates and mean-imputed values.
2. Exploratory analysis: distributions, boxplots, correlation heatmap.
3. Remove implausible records and log-transform the target (`PRICE`).
4. Split 80/20, cap outliers (IQR rule) using training-set thresholds only, check multicollinearity (VIF), fit a baseline linear regression.
5. Clean `SUBLOCALITY` into five boroughs and `TYPE` into property types, one-hot encode, and refit.
6. Compare with Random Forest and Gradient Boosting on the same split.

## Dataset

**New York Housing Market** by Nidula Elgiriyewithana, available on Kaggle: https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market (DOI: [10.34740/KAGGLE/DSV/7351086](https://doi.org/10.34740/KAGGLE/DSV/7351086)). The file contains 4,801 property listings and 17 columns. Kaggle lists its licence as "Other (specified in description)", so **the data file is not included in this repository**; please download it from Kaggle and review its terms of use.

Columns used in the analysis: `PRICE`, `BEDS`, `BATH`, `PROPERTYSQFT`, `TYPE`, `SUBLOCALITY`, `STATE` (contains the ZIP code), `LATITUDE`, `LONGITUDE`.

## How to run

1. Clone or download this repository.
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset from the Kaggle link above, create a folder called `data` next to the notebook, and save the file inside it as `NYHousing.csv` (Kaggle's file is named `NY-House-Dataset.csv`, so rename it).
4. Open the notebook with Jupyter and run all cells from top to bottom:
   ```
   jupyter notebook NY_Housing_Price_Prediction.ipynb
   ```

## Limitations

- Rows with imputed values were removed, so the models are trained on listings with complete size and bathroom data, which may not represent all listings.
- Results come from a single random train/test split, and the tree-based models were not tuned.
- Prices are listing (asking) prices, not final sale prices.
- Linear regression cannot capture interactions such as an extra square foot being worth more in Manhattan than in the Bronx.

## Next steps

- k-fold cross-validation for more reliable performance estimates
- Hyperparameter tuning for the tree-based models
- Regularised regression (Ridge, Lasso) and interaction terms
- Residual analysis, and finer location features such as ZIP code or neighbourhood

## Author

Michael Uche
