"""
Bayes (dominio: Matching Lavoro–Candidato)

Bayesian Network per stimare P(match=1 | evidenze) usando variabili discrete.

Dataset atteso: datasets/jobs-matching_supervised.csv

Output:
- results/bayes_cpds.txt
- results/bayes_test_predictions.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# pgmpy (versioni recenti)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# -----------------------
# PATH ROBUSTI
# -----------------------
SRC_DIR = Path(__file__).resolve().parent          # .../src
PROJECT_DIR = SRC_DIR.parent                      # .../ (root progetto)
DATASETS_DIR = PROJECT_DIR / "datasets"
RESULTS_DIR = PROJECT_DIR / "results"


# ---------------- Config ----------------

@dataclass
class BayesConfig:
    dataset_path: Path = DATASETS_DIR / "jobs-matching_supervised.csv"
    out_results_dir: Path = RESULTS_DIR

    test_size: float = 0.30
    random_state: int = 42

    # discretizzazione
    bins_skill: int = 3
    bins_exp: int = 3
    bins_dist: int = 3
    bins_salary_fit: int = 3

    # smoothing (Dirichlet prior)
    prior_type: str = "BDeu"
    equivalent_sample_size: int = 5


# ---------------- Model ----------------

class JobBayesNet:
    """
    BN interpretabile:
      seniority -> exp_bin
      seniority -> salary_fit_bin
      contract_type -> salary_fit_bin
      remote -> dist_bin

      skill_bin -> match
      salary_fit_bin -> match
      dist_bin -> match
      seniority -> match
      contract_type -> match
      remote -> match
    """

    def __init__(self, cfg: BayesConfig | None = None):
        self.cfg = cfg or BayesConfig()
        self.cfg.out_results_dir.mkdir(parents=True, exist_ok=True)

        self.model: DiscreteBayesianNetwork | None = None
        self.infer: VariableElimination | None = None

    # ---------- Data ----------

    def load_dataset(self) -> pd.DataFrame:
        if not self.cfg.dataset_path.exists():
            raise FileNotFoundError(
                "Dataset non trovato.\n"
                f"Sto cercando qui: {self.cfg.dataset_path.resolve()}\n\n"
                "Soluzione: metti jobs-matching_supervised.csv dentro la cartella datasets/ del progetto."
            )

        df = pd.read_csv(self.cfg.dataset_path)

        required = [
            "skill_overlap", "years_experience", "salary_offered", "distance_km",
            "seniority", "contract_type", "remote", "match"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Mancano colonne nel dataset per Bayes: " + ", ".join(missing))

        df = df.dropna(subset=required).copy()
        df["remote"] = df["remote"].astype(str).str.strip().str.lower()
        df["seniority"] = df["seniority"].astype(str).str.strip().str.lower()
        df["contract_type"] = df["contract_type"].astype(str).str.strip().str.lower()
        df["match"] = df["match"].astype(int)
        df = df[df["match"].isin([0, 1])]

        return df

    @staticmethod
    def _expected_salary(seniority: str, contract_type: str) -> float:
        base_salary = {"junior": 28000, "mid": 42000, "senior": 65000}
        contract_multiplier = {"internship": 0.55, "full_time": 1.0, "part_time": 0.75, "freelance": 1.15}
        s = base_salary.get(seniority, 42000)
        m = contract_multiplier.get(contract_type, 1.0)
        return float(s * m)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        expected = np.array([
            self._expected_salary(s, c) for s, c in zip(out["seniority"], out["contract_type"])
        ])
        salary_fit = (out["salary_offered"].to_numpy() - expected) / expected
        out["salary_fit"] = np.clip(salary_fit, -0.6, 0.8)

        return out

    @staticmethod
    def _qbin(series: pd.Series, q: int, labels_prefix: str) -> pd.Series:
        # qcut può fallire con molti duplicati; fallback a cut
        try:
            b = pd.qcut(series, q=q, duplicates="drop")
        except Exception:
            b = pd.cut(series, bins=q)
        return b.astype(str).map(lambda x: f"{labels_prefix}:{x}")

    def discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["skill_bin"] = self._qbin(out["skill_overlap"], self.cfg.bins_skill, "skill")
        out["exp_bin"] = self._qbin(out["years_experience"], self.cfg.bins_exp, "exp")
        out["dist_bin"] = self._qbin(out["distance_km"], self.cfg.bins_dist, "dist")
        out["salary_fit_bin"] = self._qbin(out["salary_fit"], self.cfg.bins_salary_fit, "salfit")

        keep = [
            "seniority", "contract_type", "remote",
            "skill_bin", "exp_bin", "dist_bin", "salary_fit_bin",
            "match"
        ]
        out = out[keep].copy()

        # pgmpy usa stati discreti: convertiamo a stringhe
        for c in ["seniority", "contract_type", "remote", "skill_bin", "exp_bin", "dist_bin", "salary_fit_bin"]:
            out[c] = out[c].astype(str)

        out["match"] = out["match"].astype(str)  # "0" / "1"
        return out

    # --- FIX pgmpy dtype inference: force categorical + state_names ---

    def to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forza ogni colonna a dtype 'category' per evitare problemi di inferenza dtype in pgmpy.
        """
        out = df.copy()
        for c in out.columns:
            out[c] = out[c].astype("category")
        return out

    @staticmethod
    def build_state_names(df: pd.DataFrame) -> dict:
        """
        Costruisce state_names per pgmpy: per ogni variabile -> lista stati possibili.
        """
        state_names = {}
        for c in df.columns:
            # ordine stabile
            cats = list(pd.Series(df[c].astype(str).unique()).sort_values())
            state_names[c] = cats
        return state_names

    # ---------- BN ----------

    def build_structure(self) -> DiscreteBayesianNetwork:
        edges = [
            ("seniority", "exp_bin"),
            ("seniority", "salary_fit_bin"),
            ("contract_type", "salary_fit_bin"),
            ("remote", "dist_bin"),

            ("skill_bin", "match"),
            ("salary_fit_bin", "match"),
            ("dist_bin", "match"),
            ("seniority", "match"),
            ("contract_type", "match"),
            ("remote", "match"),
        ]
        return DiscreteBayesianNetwork(edges)

    def fit(self, train_df: pd.DataFrame):
        # forza categorie
        train_df = self.to_categorical(train_df)

        self.model = self.build_structure()

        # evita inferenza automatica di dtype/stati
        state_names = self.build_state_names(train_df)

        self.model.fit(
            train_df,
            estimator=BayesianEstimator,
            prior_type=self.cfg.prior_type,
            equivalent_sample_size=self.cfg.equivalent_sample_size,
            state_names=state_names,
        )
        self.infer = VariableElimination(self.model)

    def save_cpds(self, path: Path):
        assert self.model is not None
        with open(path, "w", encoding="utf-8") as f:
            for cpd in self.model.get_cpds():
                f.write(str(cpd))
                f.write("\n\n")

    # ---------- Inference & Evaluation ----------

    def predict_proba_match1(self, df: pd.DataFrame) -> np.ndarray:
        """Restituisce P(match='1' | evidenze) riga per riga."""
        assert self.infer is not None

        probs = []
        for _, r in df.iterrows():
            evidence = {
                "seniority": str(r["seniority"]),
                "contract_type": str(r["contract_type"]),
                "remote": str(r["remote"]),
                "skill_bin": str(r["skill_bin"]),
                "dist_bin": str(r["dist_bin"]),
                "salary_fit_bin": str(r["salary_fit_bin"]),
                "exp_bin": str(r["exp_bin"]),
            }
            q = self.infer.query(variables=["match"], evidence=evidence, show_progress=False)

            states = list(q.state_names["match"])
            p1 = float(q.values[states.index("1")]) if "1" in states else 0.0
            probs.append(p1)

        return np.array(probs, dtype=float)

    def run(self):
        raw = self.load_dataset()
        raw = self.engineer_features(raw)
        data = self.discretize(raw)

        train_df, test_df = train_test_split(
            data,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=data["match"],
        )

        # forza categoriale anche qui (coerenza)
        train_df = self.to_categorical(train_df)
        test_df = self.to_categorical(test_df)

        self.fit(train_df)

        cpds_path = self.cfg.out_results_dir / "bayes_cpds.txt"
        self.save_cpds(cpds_path)
        print(f"CPD salvate in: {cpds_path.resolve()}")

        y_true = test_df["match"].astype(str).astype(int).to_numpy()
        y_prob = self.predict_proba_match1(test_df)

        roc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        print("\n=== Valutazione Bayes Net ===")
        print("ROC-AUC:", round(float(roc), 4))
        print("Average Precision (PR-AUC):", round(float(ap), 4))

        out_pred = test_df.copy()
        out_pred["p_match_1"] = y_prob
        out_pred_path = self.cfg.out_results_dir / "bayes_test_predictions.csv"
        out_pred.to_csv(out_pred_path, index=False)
        print(f"Predizioni salvate in: {out_pred_path.resolve()}")

        # Esempio inferenza
        example = test_df.iloc[0].to_dict()
        evidence = {k: str(example[k]) for k in ["seniority", "contract_type", "remote", "skill_bin", "dist_bin", "salary_fit_bin", "exp_bin"]}
        q = self.infer.query(variables=["match"], evidence=evidence, show_progress=False)
        print("\nEsempio inferenza P(match | evidence):")
        print("Evidence:", evidence)
        print(q)


def main():
    bn = JobBayesNet()
    bn.run()


if __name__ == "__main__":
    main()