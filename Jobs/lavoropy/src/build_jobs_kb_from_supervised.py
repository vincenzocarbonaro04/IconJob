from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- Config ----------------

@dataclass
class KBFromSupervisedConfig:
    supervised_path: Path = Path("../datasets/jobs-matching_supervised.csv")
    out_kb_path: Path = Path("../datasets/jobs-kb.csv")

    # quanti job (righe) generare nella KB
    n_jobs: int = 250

    # seed per riproducibilità
    random_state: int = 42


# ---------------- Generator ----------------

class KBFromSupervisedGenerator:
    """
    Genera un dataset simbolico (jobs-kb.csv) a partire da un dataset numerico/categoriale.

    Output columns:
    job_id, role, sector, seniority, contract_type, remote, salary_offered, location_city, required_skills
    """

    # Vocabolario controllato (accademico, riproducibile)
    ROLE_TEMPLATES = [
        ("data_scientist", "tech", ["python", "sql", "ml", "pandas", "numpy"]),
        ("machine_learning_engineer", "tech", ["python", "ml", "dl", "docker", "sql"]),
        ("backend_developer", "tech", ["python", "sql", "docker", "git", "api"]),
        ("frontend_developer", "tech", ["js", "react", "html", "css", "git"]),
        ("devops_engineer", "tech", ["docker", "kubernetes", "aws", "linux", "ci_cd"]),
        ("business_analyst", "business", ["excel", "sql", "analytics", "presentation", "requirements"]),
        ("financial_analyst", "finance", ["excel", "sql", "finance", "reporting", "risk"]),
        ("marketing_specialist", "marketing", ["seo", "analytics", "content", "ads", "communication"]),
    ]

    CITIES = ["milano", "roma", "torino", "napoli", "bologna", "firenze", "venezia", "genova", "palermo", "bari"]

    def __init__(self, cfg: KBFromSupervisedConfig | None = None):
        self.cfg = cfg or KBFromSupervisedConfig()
        self.rng = np.random.default_rng(self.cfg.random_state)

    def load_supervised(self) -> pd.DataFrame:
        if not self.cfg.supervised_path.exists():
            raise FileNotFoundError(
                f"Dataset supervisionato non trovato: {self.cfg.supervised_path.resolve()}"
            )

        df = pd.read_csv(self.cfg.supervised_path)
        required = ["salary_offered", "distance_km", "seniority", "contract_type", "remote", "skill_overlap"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Mancano colonne nel dataset supervisionato: " + ", ".join(missing))

        df = df.dropna(subset=required).copy()
        df["seniority"] = df["seniority"].astype(str).str.strip().str.lower()
        df["contract_type"] = df["contract_type"].astype(str).str.strip().str.lower()
        df["remote"] = df["remote"].astype(str).str.strip().str.lower()
        df["salary_offered"] = df["salary_offered"].astype(int)
        df["distance_km"] = df["distance_km"].astype(float)
        df["skill_overlap"] = df["skill_overlap"].astype(float)

        return df

    def _pick_role_template(self, row: pd.Series) -> tuple[str, str, list[str]]:
        """
        Scelta role/sector basata su feature osservate (sensata ma semplice):
        - remote=yes + salari alti => più probabile MLE/DevOps
        - contract internship => più probabile junior DS/FE
        - salary medio + no remote => BA/Finance/Backend
        """
        sal = row["salary_offered"]
        remote = row["remote"]
        contract = row["contract_type"]

        weights = []

        for role, sector, skills in self.ROLE_TEMPLATES:
            w = 1.0

            # bias per internship
            if contract == "internship":
                if role in ("data_scientist", "frontend_developer"):
                    w *= 2.0
                if role in ("devops_engineer", "financial_analyst"):
                    w *= 0.6

            # bias per remote
            if remote == "yes":
                if role in ("machine_learning_engineer", "devops_engineer", "frontend_developer"):
                    w *= 1.6
            else:
                if role in ("business_analyst", "financial_analyst"):
                    w *= 1.3

            # bias per salario
            if sal >= 65000:
                if role in ("machine_learning_engineer", "devops_engineer"):
                    w *= 1.9
                if role in ("marketing_specialist",):
                    w *= 0.7
            elif sal <= 28000:
                if role in ("frontend_developer", "data_scientist"):
                    w *= 1.4

            weights.append(w)

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        idx = self.rng.choice(len(self.ROLE_TEMPLATES), p=weights)
        return self.ROLE_TEMPLATES[idx]

    def _generate_required_skills(self, base_skills: list[str], row: pd.Series) -> str:
        """
        Genera una lista di skill richieste coerente con skill_overlap:
        - overlap alto => lista più corta (più “centrata”)
        - overlap basso => lista più lunga (più “esigente”/mismatch)
        """
        overlap = float(row["skill_overlap"])

        # numero skill richieste: 3..7
        # overlap alto -> 3-4, overlap basso -> 6-7
        k = int(np.clip(round(7 - 4 * overlap), 3, 7))

        # prendiamo k skill da base + eventuali extra
        extras = ["git", "communication", "problem_solving", "teamwork", "english", "testing", "agile"]
        pool = list(dict.fromkeys(base_skills + extras))  # unique, stable order

        skills = list(self.rng.choice(pool, size=min(k, len(pool)), replace=False))
        # normalizza con underscore (già lo sono)
        return ",".join(skills)

    def _assign_city(self, row: pd.Series) -> str:
        """
        Non abbiamo la città nel dataset supervisionato, quindi la generiamo in modo sensato:
        - se remote=yes: città a scelta (non vincola)
        - se remote=no: preferenza per città grandi (Milano/Roma/Torino/Bologna)
        """
        remote = row["remote"]
        if remote == "yes":
            return str(self.rng.choice(self.CITIES))
        big = ["milano", "roma", "torino", "bologna"]
        return str(self.rng.choice(big))

    def generate(self) -> pd.DataFrame:
        df = self.load_supervised()

        # campiona righe del dataset supervisionato come "base" per job
        n = min(self.cfg.n_jobs, len(df))
        sampled = df.sample(n=n, replace=False, random_state=self.cfg.random_state).reset_index(drop=True)

        rows = []
        for i, r in sampled.iterrows():
            job_id = f"job_{i+1}"

            role, sector, base_skills = self._pick_role_template(r)
            city = self._assign_city(r)
            req_skills = self._generate_required_skills(base_skills, r)

            rows.append(
                {
                    "job_id": job_id,
                    "role": role,
                    "sector": sector,
                    "seniority": r["seniority"],
                    "contract_type": r["contract_type"],
                    "remote": r["remote"],
                    "salary_offered": int(r["salary_offered"]),
                    "location_city": city,
                    "required_skills": req_skills,
                }
            )

        kb = pd.DataFrame(rows)

        # piccola pulizia: valori ammessi
        kb["remote"] = kb["remote"].where(kb["remote"].isin(["yes", "no"]), "no")
        kb["seniority"] = kb["seniority"].where(kb["seniority"].isin(["junior", "mid", "senior"]), "mid")
        return kb

    def save(self, kb: pd.DataFrame) -> Path:
        self.cfg.out_kb_path.parent.mkdir(parents=True, exist_ok=True)
        kb.to_csv(self.cfg.out_kb_path, index=False)
        return self.cfg.out_kb_path

    def run(self):
        kb = self.generate()
        out = self.save(kb)
        print(f"Creato: {out.resolve()}")
        print("Righe:", len(kb))
        print("\nEsempio:")
        print(kb.head(5))


def main():
    gen = KBFromSupervisedGenerator()
    gen.run()


if __name__ == "__main__":
    main()