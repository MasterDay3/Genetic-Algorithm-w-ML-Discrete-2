"""
dataset preparing and unpacking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# FILENAME = "datasets/heart.csv"
# FILENAME = "datasets/diabetes.csv"
# FILENAME = "datasets/hospital_readmission.csv"
FILENAME = "datasets/working_hours.csv"


print(f"Loading dataset {FILENAME}")
df = pd.read_csv(FILENAME)
df = df.drop_duplicates()
df = df.sample(10000)


def convert_to_numeric(df):
    """Allows to work with dataset, converts all features into digits"""
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.isna().sum() <= df[col].isna().sum():
                df[col] = converted
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median(numeric_only=True))

    return df


print(df.columns)
# if FILENAME == "datasets/working_hours.csv":
#     leakage_features = [
#         "Flow Bytes/s",
#         " Flow Packets/s",
#         " Flow Duration",
#         " Flow IAT Mean",
#         " Flow IAT Std",
#         " Flow IAT Max",
#         " Flow IAT Min",
#         "Fwd IAT Total",
#         " Fwd IAT Mean",
#         " Fwd IAT Std",
#         " Fwd IAT Max",
#         " Fwd IAT Min",
#         "Bwd IAT Total",
#         " Bwd IAT Mean",
#         " Bwd IAT Std",
#         " Bwd IAT Max",
#         " Bwd IAT Min",
#         "Active Mean",
#         " Active Std",
#         " Active Max",
#         " Active Min",
#         "Idle Mean",
#         " Idle Std",
#         " Idle Max",
#         " Idle Min",
#         "Init_Win_bytes_forward",
#         " Init_Win_bytes_backward",
#         "Destination Port",
#     ]

# df = df.drop(columns=[c for c in leakage_features if c in df.columns])
if FILENAME == "datasets/hospital_readmission.csv":
    df = df.drop(columns="readmission_risk_score")

print(df.columns.tolist())

df = convert_to_numeric(df)
df = df.dropna()
target_name = df.columns[-1]
y = df[target_name]
X = df.drop(columns=[target_name])
X = pd.get_dummies(X, drop_first=True)

features = df.columns
feature_types = dict(df.dtypes)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

print(f"Dataset: {df.shape}")
