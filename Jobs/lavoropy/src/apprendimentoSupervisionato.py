"""
Apprendimento supervisionato (dominio: matching Lavoro–Candidato).

Questo modulo sostituisce il precedente "ranking giochi" e implementa un
classificatore binario che predice se una coppia (candidato, offerta) è un buon
match (colonna target: `match`).

Dataset atteso: ../datasets/jobs-matching_supervised.csv
Colonne minime:
- Numeriche: skill_overlap, years_experience, salary_offered, distance_km
- Categoriche: seniority, contract_type, remote
- Target: match (0/1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SupervisedJobConfig:
    """Configurazione del task supervisionato."""

    dataset_path: Path = Path("../datasets/jobs-matching_supervised.csv")
    target_col: str = "match"

    numerical_features: Tuple[str, ...] = (
        "skill_overlap",
        "years_experience",
        "salary_offered",
        "distance_km",
    )
    categorical_features: Tuple[str, ...] = (
        "seniority",
        "contract_type",
        "remote",
    )

    test_size: float = 0.30
    random_state: int = 42
    cv_splits: int = 5

    # dove salvare le immagini (rispetta la struttura esistente del progetto)
    out_img_dir: Path = Path("../img/Apprendimento_supervisionato")


class SupervisedJobMatcher:
    """Pipeline end-to-end: carica dati, addestra, ottimizza e valuta."""

    def __init__(self, cfg: SupervisedJobConfig | None = None):
        self.cfg = cfg or SupervisedJobConfig()
        self.cfg.out_img_dir.mkdir(parents=True, exist_ok=True)

        self._preprocessor = self._build_preprocessor()

    # ---------- Data ----------

    def load_dataset(self) -> pd.DataFrame:
        if not self.cfg.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset non trovato: {self.cfg.dataset_path.resolve()}\n"
                "Suggerimento: metti il CSV in datasets/ con nome 'jobs-matching_supervised.csv'."
            )

        df = pd.read_csv(self.cfg.dataset_path)

        required = set(self.cfg.numerical_features) | set(self.cfg.categorical_features) | {self.cfg.target_col}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "Il dataset non contiene tutte le colonne richieste. Mancano: " + ", ".join(missing)
            )

        # pulizia minima
        df = df.dropna(subset=list(required)).copy()

        # normalizza target a 0/1
        df[self.cfg.target_col] = df[self.cfg.target_col].astype(int)
        df = df[df[self.cfg.target_col].isin([0, 1])]

        return df

    def train_test_split(self, df: pd.DataFrame):
        X = df[list(self.cfg.numerical_features) + list(self.cfg.categorical_features)]
        y = df[self.cfg.target_col]

        return train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )

    # ---------- Pipeline ----------

    def _build_preprocessor(self) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
        cat_pipe = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, list(self.cfg.numerical_features)),
                ("cat", cat_pipe, list(self.cfg.categorical_features)),
            ]
        )

    def build_model_zoo(self) -> Dict[str, Tuple[object, Dict]]:
        """Modelli + griglie di iperparametri (tutti compatibili con sklearn)."""
        return {
            "RandomForest": (
                RandomForestClassifier(random_state=self.cfg.random_state),
                {
                    "clf__n_estimators": [200, 400],
                    "clf__max_depth": [None, 8, 16],
                    "clf__min_samples_split": [2, 5],
                },
            ),
            "SVM-RBF": (
                SVC(probability=True, random_state=self.cfg.random_state),
                {
                    "clf__C": [0.5, 1, 5],
                    "clf__gamma": ["scale", 0.1, 0.01],
                },
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=self.cfg.random_state),
                {
                    "clf__max_depth": [None, 5, 10, 20],
                    "clf__min_samples_split": [2, 5, 10],
                },
            ),
            "KNN": (
                KNeighborsClassifier(),
                {
                    "clf__n_neighbors": [5, 11, 21],
                    "clf__weights": ["uniform", "distance"],
                },
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=self.cfg.random_state),
                {
                    "clf__n_estimators": [150, 300],
                    "clf__learning_rate": [0.05, 0.1],
                    "clf__max_depth": [2, 3],
                },
            ),
        }

    def make_pipeline(self, clf) -> Pipeline:
        return Pipeline(steps=[("prep", self._preprocessor), ("clf", clf)])

    # ---------- Training + Evaluation ----------

    @staticmethod
    def _safe_predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
        """Restituisce proba classe positiva. Compatibile con modelli senza predict_proba."""
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X)[:, 1]
        # fallback: decision_function -> logistic-ish scaling
        if hasattr(pipeline, "decision_function"):
            s = pipeline.decision_function(X)
            return 1 / (1 + np.exp(-s))
        # ultimo fallback
        return pipeline.predict(X).astype(float)

    def plot_learning_curve(self, pipeline: Pipeline, X, y, model_name: str):
        cv = StratifiedKFold(n_splits=self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state)
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X,
            y,
            cv=cv,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="roc_auc",
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 7))
        plt.plot(train_sizes, train_mean, marker="o", label="Training ROC-AUC")
        plt.plot(train_sizes, test_mean, marker="o", label="CV ROC-AUC")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Training size")
        plt.ylabel("ROC-AUC")
        plt.grid(True)
        plt.legend(loc="best")
        out = self.cfg.out_img_dir / f"{model_name}_learningCurve.png"
        plt.savefig(out, bbox_inches="tight")
        plt.show()

    def plot_roc(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {model_name}")
        plt.grid(True)
        plt.legend(loc="best")
        out = self.cfg.out_img_dir / f"{model_name}_ROC.png"
        plt.savefig(out, bbox_inches="tight")
        plt.show()

    @staticmethod
    def gmap_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """GMAP coerente per binario: geometric mean di AP(pos) e AP(neg)."""
        ap_pos = average_precision_score(y_true, y_prob)
        ap_neg = average_precision_score(1 - y_true, 1 - y_prob)
        return float(np.sqrt(max(ap_pos, 1e-12) * max(ap_neg, 1e-12)))

    def train_with_grid_search(
        self,
        base_clf,
        param_grid: Dict,
        X_train,
        y_train,
    ) -> GridSearchCV:
        pipeline = self.make_pipeline(base_clf)
        cv = StratifiedKFold(n_splits=self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state)

        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        return gs

    def evaluate(self, pipeline: Pipeline, X_test, y_test, model_name: str) -> Dict[str, float]:
        y_pred = pipeline.predict(X_test)
        y_prob = self._safe_predict_proba(pipeline, X_test)

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        gmap = self.gmap_binary(y_test.to_numpy(), y_prob)

        print("\n=== Risultati:", model_name, "===")
        print("Accuracy:", round(acc, 4))
        print("ROC-AUC:", round(roc_auc, 4))
        print("Average Precision (PR-AUC):", round(ap, 4))
        print("GMAP (binary):", round(gmap, 4))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

        self.plot_roc(y_test, y_prob, model_name)
        self.plot_learning_curve(pipeline, X_test, y_test, model_name)

        return {"accuracy": acc, "roc_auc": roc_auc, "average_precision": ap, "gmap": gmap}


def main():
    matcher = SupervisedJobMatcher()
    df = matcher.load_dataset()
    X_train, X_test, y_train, y_test = matcher.train_test_split(df)

    zoo = matcher.build_model_zoo()

    print("\nApprendimento supervisionato - Matching Lavoro–Candidato")
    print("Dataset:", matcher.cfg.dataset_path)
    print("Modelli disponibili:")
    for i, k in enumerate(zoo.keys(), start=1):
        print(f"  {i}. {k}")

    try:
        choice = int(input("\nSeleziona modello (numero): "))
    except Exception:
        choice = 1

    model_name = list(zoo.keys())[max(0, min(choice - 1, len(zoo) - 1))]
    base_clf, grid = zoo[model_name]

    print(f"\n> GridSearchCV su {model_name}...")
    gs = matcher.train_with_grid_search(base_clf, grid, X_train, y_train)

    print("\nMigliori iperparametri:")
    print(gs.best_params_)

    best_pipe = gs.best_estimator_
    matcher.evaluate(best_pipe, X_test, y_test, model_name)


if __name__ == "__main__":
    main()