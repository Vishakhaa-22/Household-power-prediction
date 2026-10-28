# Household Power Consumption Prediction

A machine learning project that predicts a household's **total electric power (kW)** from appliance
sub-meter readings, line voltage and the date/hour. It compares three algorithms
(**Linear Regression, Decision Tree, Random Forest**) and includes a **Streamlit web app** where a user
enters values and gets predictions instantly.

> AIML team project | Gandhinagar University | B.tech-C.E

---

## Problem statement

Given the readings of three appliance sub-meters, the voltage, and the time of day, estimate the total
active power the household is drawing at that moment.

- **Type:** supervised regression
- **Target:** `Global_active_power` (kW)
- **Inputs:** `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`, `Voltage`, hour, day of week, weekend flag, month

### A note on data leakage
An early version of the model used `Global_intensity` (electric current) as an input and scored R² = 0.998.
That is misleading: power is voltage × current, so the model was effectively handed the answer.
We removed `Global_intensity` and `Global_reactive_power` and kept only inputs that a user could realistically supply.

The three sub-meters measure *parts* of the household's consumption (kitchen, laundry room, water heater and
air-conditioner), so the task is best described as **estimating total load from appliance readings**.

---

## Dataset

[UCI - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

- One household in France, December 2006 to November 2010
- About 2.07 million one-minute readings (about 2.05 million after removing rows with missing values)
- The raw file (about 133 MB) is **not** included in this repository because of GitHub's size limit

To rerun the notebook, download the dataset and place `household_power_consumption.txt` in `data/raw/`.

---

## Method

1. Cleaning: parse date and time into a datetime index, drop rows with missing values.
2. Features: the four meter/voltage inputs, plus cyclical hour (sin/cos), day of week, weekend flag and month.
3. Split: chronological 80/20 (train on the earlier period, test on the later period).
4. Models: Linear Regression, Decision Tree, Random Forest, compared with a "predict the mean" baseline.
5. Evaluation: MAE, RMSE and R² on the test period, plus 5-fold cross-validation as a stability check.

---

## Results

Test period: Feb to Nov 2010 (data the models never saw during training).

| Model | MAE (kW) | RMSE (kW) | R² |
| Random Forest | 0.215 | 0.350| 0.848 |
| Decision Tree | 0.231 | 0.369 | 0.830 |
| Linear Regression | 0.297 | 0.396 | 0.805 |
| Baseline (always the mean) | 0.726 | 0.901 | -0.011 |

All three models clearly beat the baseline, and Random Forest performs best. Exact numbers can vary slightly
between runs; see `models/model_comparison.csv` for the values produced by the notebook.

---

## Project structure

```
├── app.py                         # Streamlit web app
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_realtime_prediction.ipynb   # feature engineering, training, evaluation, saving models
├── models/
│   ├── models.pkl                 # the three trained models
│   └── model_comparison.csv
└── data/
    ├── raw/                       # NOT in the repo (download the dataset here)
    └── processed/
        └── live_sample.csv        # a week of real readings used by the app's demo
```


## How to run

**1. Clone the repository**
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

**2. Install the requirements**
```bash
python -m pip install -r requirements.txt
```

**3. Start the app**
```bash
python -m streamlit run app.py
```
The app opens at `http://localhost:8501`. The trained models are included, so no retraining is needed.

**Optional: retrain the models**
Download the dataset into `data/raw/`, open `notebooks/02_realtime_prediction.ipynb` and run all cells.
The last cell regenerates everything in `models/` and `data/processed/`.

---

## The app

- Enter the three sub-meter readings, the voltage, a date and an hour.
- All three models predict the total power (kW) at once, side by side.
- Compare model performance and see how each one behaves on real readings.

Live demo: [add your Streamlit Cloud link here, if deployed]

---

## Limitations

- Data comes from a single household in France, so the models may not transfer to other homes.
- The sub-meters cover only part of the household's load.
- The models were trained on 2006-2010 usage patterns.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Streamlit, joblib

## Team

| Name | Enrollment No. | Role |
| [Vishakha Nebhanani] | [240101005066] | [ ] |
| [Henny Lathiya] | [ 240101005051] | [ ] |
| [Name 3] | [ ] | [ ] |
| [Name 4] | [ ] | [ ] |
| [Name 5] | [ ] | [ ] |
| [Name 6] | [ ] | [ ] |

Guide: [Rashmi h]
