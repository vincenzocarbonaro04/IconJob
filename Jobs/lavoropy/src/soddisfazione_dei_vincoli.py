"""
Risoluzione di un CSP (dominio: Matching Lavoro–Candidato)

Obiettivo: costruire una short-list di 10 offerte "compatibili" per un candidato,
massimizzando coerenza e rispettando vincoli (hard+soft).

Dataset atteso: ../datasets/jobs-matching_supervised.csv
Colonne usate:
- skill_overlap (0..1) [proxy coerenza skill]
- salary_offered (int)
- distance_km (float)
- seniority (junior/mid/senior)
- contract_type (internship/full_time/part_time/freelance)
- remote (yes/no)

Algoritmi:
- Random Walk
- Simulated Annealing
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import random
import time


# ---------------- Config & Domain ----------------

@dataclass(frozen=True)
class CSPJobConfig:
    dataset_path: Path = Path("../datasets/jobs-matching_supervised.csv")

    shortlist_size: int = 10

    # Vincoli candidato (default "accademici": modificabili da input)
    min_salary: int = 35000
    remote_ok: bool = True
    max_distance_km_if_not_remote: float = 60.0

    # Vincoli di qualità/diversità (accademicamente difendibili)
    min_avg_skill_overlap: float = 0.55
    max_same_seniority: int = 4
    max_same_contract: int = 5

    # Per simulated annealing
    max_iter: int = 2000
    temp0: float = 800.0
    alpha: float = 0.995

    random_state: int = 42

    out_results_dir: Path = Path("../results")


class CSPJobShortlistSolver:
    """
    Risolve un CSP di selezione: scegliere N offerte dal dataset
    minimizzando violazioni di vincoli (penalty).
    """

    def __init__(self, cfg: CSPJobConfig | None = None):
        self.cfg = cfg or CSPJobConfig()
        self.cfg.out_results_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)

    # ---------- Data ----------

    def load_dataset(self) -> pd.DataFrame:
        if not self.cfg.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset non trovato: {self.cfg.dataset_path.resolve()}\n"
                "Metti il CSV in ../datasets/jobs-matching_supervised.csv"
            )

        df = pd.read_csv(self.cfg.dataset_path)

        required = [
            "skill_overlap", "salary_offered", "distance_km",
            "seniority", "contract_type", "remote"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Mancano colonne nel dataset per CSP: " + ", ".join(missing))

        df = df.dropna(subset=required).copy()

        # Normalizza formati
        df["remote"] = df["remote"].astype(str).str.strip().str.lower()
        df["seniority"] = df["seniority"].astype(str).str.strip().str.lower()
        df["contract_type"] = df["contract_type"].astype(str).str.strip().str.lower()

        df["salary_offered"] = df["salary_offered"].astype(int)
        df["distance_km"] = df["distance_km"].astype(float)
        df["skill_overlap"] = df["skill_overlap"].astype(float)

        # crea un job_id comodo (se non esiste)
        if "job_id" not in df.columns:
            df = df.reset_index(drop=True)
            df["job_id"] = ["job_" + str(i) for i in range(len(df))]

        return df

    # ---------- CSP constraints ----------

    def _hard_constraints_ok(self, sol: pd.DataFrame) -> bool:
        # 1) cardinalità esatta
        if len(sol) != self.cfg.shortlist_size:
            return False

        # 2) salario minimo
        if (sol["salary_offered"] < self.cfg.min_salary).any():
            return False

        # 3) distanza se non remote (remote offerta = no) e candidato non remote_ok
        # - Se candidato remote_ok=False: accetta solo remote=no e entro max_distance
        if not self.cfg.remote_ok:
            if (sol["remote"] != "no").any():
                return False
            if (sol["distance_km"] > self.cfg.max_distance_km_if_not_remote).any():
                return False
        else:
            # candidato remote_ok=True:
            # per offerte non-remote imponi max distanza
            mask_not_remote = sol["remote"] == "no"
            if (sol.loc[mask_not_remote, "distance_km"] > self.cfg.max_distance_km_if_not_remote).any():
                return False

        return True

    def penalty(self, sol: pd.DataFrame) -> float:
        """
        Funzione obiettivo: 0 = soluzione perfetta.
        Penalizza violazioni hard (molto) e soft (meno).
        """
        penalty = 0.0

        # hard: cardinalità
        if len(sol) != self.cfg.shortlist_size:
            penalty += 1000.0 * abs(self.cfg.shortlist_size - len(sol))

        # hard: salario minimo
        below = (self.cfg.min_salary - sol["salary_offered"]).clip(lower=0)
        penalty += 2.0 * float(below.sum())  # scala su euro (penalità importante)

        # hard: distanza secondo regole remote
        if not self.cfg.remote_ok:
            # remote non permesso
            penalty += 300.0 * float((sol["remote"] != "no").sum())
            over = (sol["distance_km"] - self.cfg.max_distance_km_if_not_remote).clip(lower=0)
            penalty += 50.0 * float(over.sum())
        else:
            # remote permesso, ma non-remote entro max distanza
            mask_not_remote = sol["remote"] == "no"
            over = (sol.loc[mask_not_remote, "distance_km"] - self.cfg.max_distance_km_if_not_remote).clip(lower=0)
            penalty += 50.0 * float(over.sum())

        # soft: coerenza skill media
        avg_overlap = float(sol["skill_overlap"].mean()) if len(sol) else 0.0
        if avg_overlap < self.cfg.min_avg_skill_overlap:
            penalty += 800.0 * (self.cfg.min_avg_skill_overlap - avg_overlap)

        # soft: diversità seniority
        counts_sen = sol["seniority"].value_counts()
        excess_sen = (counts_sen - self.cfg.max_same_seniority).clip(lower=0)
        penalty += 120.0 * float(excess_sen.sum())

        # soft: diversità contract
        counts_con = sol["contract_type"].value_counts()
        excess_con = (counts_con - self.cfg.max_same_contract).clip(lower=0)
        penalty += 80.0 * float(excess_con.sum())

        # soft: preferisci salari più alti (non è vincolo, ma obiettivo)
        # (penalità negativa = bonus) -> piccolo, così non domina
        penalty -= 0.0005 * float(sol["salary_offered"].sum())

        return float(penalty)

    def constraints_satisfied(self, sol: pd.DataFrame) -> bool:
        return self._hard_constraints_ok(sol) and self.penalty(sol) <= 0.0001

    # ---------- Search algorithms ----------

    def random_walk(self, df: pd.DataFrame, max_iter: int = 2000) -> pd.DataFrame:
        best_sol = None
        best_pen = float("inf")

        for _ in range(max_iter):
            sol = df.sample(n=self.cfg.shortlist_size, replace=False)
            pen = self.penalty(sol)

            if pen < best_pen:
                best_pen = pen
                best_sol = sol

            if best_pen <= 0.0001 and self._hard_constraints_ok(best_sol):
                break

        return best_sol

    def simulated_annealing(
        self,
        df: pd.DataFrame,
        max_iter: int | None = None,
        temp0: float | None = None,
        alpha: float | None = None
    ) -> pd.DataFrame:
        max_iter = max_iter or self.cfg.max_iter
        temp = temp0 or self.cfg.temp0
        alpha = alpha or self.cfg.alpha

        current = df.sample(n=self.cfg.shortlist_size, replace=False)
        current_pen = self.penalty(current)

        best = current
        best_pen = current_pen

        for _ in range(max_iter):
            temp *= alpha
            if temp <= 1e-9:
                break

            # neighbor: cambia 1 elemento della shortlist
            new = current.copy()

            # rimuovi 1 riga a caso
            drop_idx = random.choice(list(new.index))
            new = new.drop(index=drop_idx)

            # aggiungi 1 offerta a caso non già presente
            remaining = df.loc[~df.index.isin(new.index)]
            add_row = remaining.sample(n=1, replace=False)
            new = pd.concat([new, add_row], axis=0)

            new_pen = self.penalty(new)
            delta = new_pen - current_pen

            # accetta se migliore o con prob. exp(-delta/temp)
            if delta < 0 or np.exp(-delta / temp) > random.random():
                current = new
                current_pen = new_pen

                if new_pen < best_pen:
                    best = new
                    best_pen = new_pen

            # early stop se molto buono
            if best_pen <= 0.0001 and self._hard_constraints_ok(best):
                break

        return best

    # ---------- I/O & reporting ----------

    def save_solution(self, sol: pd.DataFrame, filename: str):
        out = self.cfg.out_results_dir / filename
        sol_sorted = sol.sort_values(by=["salary_offered"], ascending=False).reset_index(drop=True)
        sol_sorted.to_csv(out, index=False)
        print(f"Risultato salvato in: {out.resolve()}")

    def print_solution_summary(self, sol: pd.DataFrame, name: str):
        print(f"\n=== {name} ===")
        print("Penalty:", round(self.penalty(sol), 4))
        print("Hard constraints OK:", self._hard_constraints_ok(sol))
        print("Avg skill_overlap:", round(float(sol["skill_overlap"].mean()), 3))
        print("Avg salary:", round(float(sol["salary_offered"].mean()), 1))
        print("Avg distance_km:", round(float(sol["distance_km"].mean()), 1))
        print("\nSeniority counts:\n", sol["seniority"].value_counts())
        print("\nContract counts:\n", sol["contract_type"].value_counts())
        print("\nRemote counts:\n", sol["remote"].value_counts())

    # ---------- CLI ----------

    def run_benchmark(self, df: pd.DataFrame, method: str, runs: int = 10):
        times = []
        pens = []

        for _ in range(runs):
            t0 = time.time()
            if method == "rw":
                sol = self.random_walk(df, max_iter=2000)
            else:
                sol = self.simulated_annealing(df)
            t1 = time.time()

            times.append(t1 - t0)
            pens.append(self.penalty(sol))

        print(f"\nBenchmark ({method}) su {runs} run:")
        print("Tempo medio:", round(float(np.mean(times)), 4), "s")
        print("Penalty media:", round(float(np.mean(pens)), 4))

    def main(self):
        df = self.load_dataset()

        print("\n=== CSP: Short-list di offerte (Matching Lavoro–Candidato) ===")
        print("Dataset:", self.cfg.dataset_path)
        print(f"Shortlist size: {self.cfg.shortlist_size}")

        # Parametri candidato
        try:
            min_salary = input(f"Salario minimo (default {self.cfg.min_salary}): ").strip()
            if min_salary:
                object.__setattr__(self.cfg, "min_salary", int(min_salary))  # dataclass frozen workaround not allowed
        except Exception:
            pass

        # menu
        while True:
            print("\nScegli un metodo di ottimizzazione:")
            print("1) Random Walk")
            print("2) Simulated Annealing")
            print("3) Benchmark (10 run) Random Walk")
            print("4) Benchmark (10 run) Simulated Annealing")
            print("0) Esci")

            scelta = input("Scelta: ").strip()

            if scelta == "0":
                break

            if scelta == "1":
                sol = self.random_walk(df, max_iter=2000)
                self.print_solution_summary(sol, "Random Walk solution")
                self.save_solution(sol, "csp_random_walk_shortlist.csv")

            elif scelta == "2":
                sol = self.simulated_annealing(df)
                self.print_solution_summary(sol, "Simulated Annealing solution")
                self.save_solution(sol, "csp_simulated_annealing_shortlist.csv")

            elif scelta == "3":
                self.run_benchmark(df, method="rw", runs=10)

            elif scelta == "4":
                self.run_benchmark(df, method="sa", runs=10)

            else:
                print("Scelta non valida.")


def main():
    solver = CSPJobShortlistSolver()
    solver.main()


if __name__ == "__main__":
    main()