from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DEFAULT_SCORING = "roc_auc"
# models.py

# DEFAULT_MODEL = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
DEFAULT_MODEL = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))


def evaluate_baseline(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    return acc, f1, auc


log_reg = DEFAULT_MODEL


def fitness_function(
    chromosome: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    penalty: float = 0.05,
    cv: int = 5,
    scoring: str = DEFAULT_SCORING,
) -> float:
    selected_indices = np.where(chromosome == 1)[0]

    if len(selected_indices) == 0:
        return 0.0

    X_subset = X_train[:, selected_indices]
    scores = cross_val_score(model, X_subset, y_train, cv=cv, scoring=scoring)
    feature_ratio = len(selected_indices) / len(chromosome)

    return scores.mean() - penalty * feature_ratio
