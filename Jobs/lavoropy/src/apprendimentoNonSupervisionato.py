"""
Apprendimento non supervisionato (dominio: matching Lavoro–Candidato).

Clustering di istanze (coppie candidato-offerta) per individuare segmenti di mercato
e anomalie (outlier) in base a feature strutturate.

Dataset atteso: ../datasets/jobs-matching_supervised.csv
Colonne minime:
- Numeriche: skill_overlap, years_experience, salary_offered, distance_km
- Categoriche: seniority, contract_type, remote
- (opzionale) target: match -> ignorato in questo modulo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class UnsupervisedJobConfig:
    dataset_path: Path = Path("../datasets/jobs-matching_supervised.csv")

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

    random_state: int = 42
    k_min: int = 2
    k_max: int = 10

    out_img_dir: Path = Path("../img/Apprendimento_non_supervisionato")
    out_results_dir: Path = Path("../results")

    anomaly_percentile: float = 95.0


class UnsupervisedJobClusterer:
    def __init__(self, cfg: UnsupervisedJobConfig | None = None):
        self.cfg = cfg or UnsupervisedJobConfig()
        self.cfg.out_img_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.out_results_dir.mkdir(parents=True, exist_ok=True)

        self._preprocessor = self._build_preprocessor()

    # ---------- Data ----------

    def load_dataset(self) -> pd.DataFrame:
        if not self.cfg.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset non trovato: {self.cfg.dataset_path.resolve()}\n"
                "Suggerimento: metti il CSV in datasets/ con nome 'jobs-matching_supervised.csv'."
            )

        df = pd.read_csv(self.cfg.dataset_path)

        required = set(self.cfg.numerical_features) | set(self.cfg.categorical_features)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Mancano colonne richieste nel dataset: " + ", ".join(missing))

        df = df.dropna(subset=list(required)).copy()
        return df

    def _build_preprocessor(self) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
        cat_pipe = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, list(self.cfg.numerical_features)),
                ("cat", cat_pipe, list(self.cfg.categorical_features)),
            ]
        )

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[list(self.cfg.numerical_features) + list(self.cfg.categorical_features)]

    # ---------- Model selection ----------

    def compute_elbow_and_silhouette(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola inertia (elbow) e silhouette per K in [k_min, k_max].
        Silhouette è definito solo per K>=2.
        """
        inertia: List[float] = []
        silhouettes: List[float] = []
        ks = list(range(self.cfg.k_min, self.cfg.k_max + 1))

        X_tr = self._preprocessor.fit_transform(X)

        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.cfg.random_state, n_init="auto")
            labels = km.fit_predict(X_tr)
            inertia.append(float(km.inertia_))

            # Silhouette score (più alto è meglio)
            sil = silhouette_score(X_tr, labels)
            silhouettes.append(float(sil))

        res = pd.DataFrame({"k": ks, "inertia": inertia, "silhouette": silhouettes})
        return res

    def plot_elbow(self, scores: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.plot(scores["k"], scores["inertia"], marker="o")
        plt.xlabel("Numero di cluster (K)")
        plt.ylabel("Inertia")
        plt.title("Elbow curve (KMeans) - Matching Lavoro–Candidato")
        plt.grid(True)
        out = self.cfg.out_img_dir / "elbow_curve_jobs.png"
        plt.savefig(out, bbox_inches="tight")
        plt.show()

    def plot_silhouette(self, scores: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.plot(scores["k"], scores["silhouette"], marker="o")
        plt.xlabel("Numero di cluster (K)")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score (KMeans) - Matching Lavoro–Candidato")
        plt.grid(True)
        out = self.cfg.out_img_dir / "silhouette_curve_jobs.png"
        plt.savefig(out, bbox_inches="tight")
        plt.show()

    def choose_k(self, scores: pd.DataFrame) -> int:
        """
        Scelta accademicamente difendibile: K che massimizza silhouette.
        (In relazione potete discutere anche elbow come supporto.)
        """
        best_row = scores.loc[scores["silhouette"].idxmax()]
        return int(best_row["k"])

    # ---------- Fit + analysis ----------

    def fit_kmeans(self, X: pd.DataFrame, k: int) -> Pipeline:
        pipe = Pipeline(
            steps=[
                ("prep", self._preprocessor),
                ("kmeans", KMeans(n_clusters=k, random_state=self.cfg.random_state, n_init="auto")),
            ]
        )
        pipe.fit(X)
        return pipe

    @staticmethod
    def _min_distance_to_centroid(X_tr: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # distanza euclidea al centroide più vicino
        d = np.linalg.norm(X_tr[:, None, :] - centroids[None, :, :], axis=2)
        return d.min(axis=1)

    def detect_anomalies(self, pipe: Pipeline, X: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        X_tr = pipe.named_steps["prep"].transform(X)
        centroids = pipe.named_steps["kmeans"].cluster_centers_
        distances = self._min_distance_to_centroid(X_tr, centroids)

        threshold = np.percentile(distances, self.cfg.anomaly_percentile)
        outlier_mask = distances > threshold

        out_df = df_original.copy()
        out_df["cluster"] = pipe.named_steps["kmeans"].labels_
        out_df["distance_from_centroid"] = distances
        anomalies = out_df.loc[outlier_mask].sort_values("distance_from_centroid", ascending=False)

        out_path = self.cfg.out_results_dir / "anomalies_jobs.csv"
        anomalies.to_csv(out_path, index=False)

        return anomalies

    def plot_clusters_pca(self, pipe: Pipeline, X: pd.DataFrame, labels: np.ndarray):
        X_tr = pipe.named_steps["prep"].transform(X)

        pca = PCA(n_components=2, random_state=self.cfg.random_state)
        X_2d = pca.fit_transform(X_tr)

        plt.figure(figsize=(11, 8))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=18)
        plt.title("KMeans clustering (PCA 2D) - Matching Lavoro–Candidato")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        out = self.cfg.out_img_dir / "kmeans_pca_jobs.png"
        plt.savefig(out, bbox_inches="tight")
        plt.show()

    def print_cluster_summary(self, df: pd.DataFrame, label_col: str = "cluster"):
        print("\nDistribuzione cluster:")
        print(df[label_col].value_counts().sort_index())

        print("\nMedie per cluster (numeriche):")
        print(df.groupby(label_col)[list(self.cfg.numerical_features)].mean().round(3))

        print("\nModa per cluster (categoriche):")
        for c in self.cfg.categorical_features:
            mode_by_cluster = df.groupby(label_col)[c].agg(lambda x: x.value_counts().index[0])
            print(f"- {c}:\n{mode_by_cluster}\n")

    # ---------- Run ----------

    def run(self):
        df = self.load_dataset()
        X = self._X(df)

        scores = self.compute_elbow_and_silhouette(X)
        print("\nK selection table:\n", scores)

        self.plot_elbow(scores)
        self.plot_silhouette(scores)

        k = self.choose_k(scores)
        print(f"\nK scelto (max silhouette): {k}")

        pipe = self.fit_kmeans(X, k)
        labels = pipe.named_steps["kmeans"].labels_

        df_out = df.copy()
        df_out["cluster"] = labels

        self.plot_clusters_pca(pipe, X, labels)
        self.print_cluster_summary(df_out, "cluster")

        anomalies = self.detect_anomalies(pipe, X, df)
        print(f"\nNumero anomalie identificate (>{self.cfg.anomaly_percentile}° percentile): {len(anomalies)}")
        print("Anomalie salvate in: ../results/anomalies_jobs.csv")


def main():
    clusterer = UnsupervisedJobClusterer()
    clusterer.run()


if __name__ == "__main__":
    main()