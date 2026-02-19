"""
Rappresentazione e ragionamento - Creazione Knowledge Base (Prolog)
Dominio: Matching Lavoro–Candidato

Genera un file Prolog: jobs_kb.pl

Dataset atteso: datasets/jobs-kb.csv (path risolto in modo robusto)

Schema minimo (colonne):
- job_id (string/int)
- role (string)                es. "data_scientist"
- sector (string)              es. "tech", "finance"
- seniority (string)           "junior|mid|senior"
- contract_type (string)       "internship|full_time|part_time|freelance"
- remote (string)              "yes|no"
- salary_offered (int)
- location_city (string)       es. "milano"
- required_skills (string)     lista skill separate da virgola, es: "python,ml,sql"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


def norm_atom(s: str) -> str:
    """Normalizza una stringa in un atomo Prolog sicuro."""
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("_")
    return s if s else "unknown"


def split_skills(raw: str) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    items = [norm_atom(x) for x in str(raw).split(",")]
    return [x for x in items if x and x != "unknown"]


# -----------------------
# PATH ROBUSTI
# -----------------------
SRC_DIR = Path(__file__).resolve().parent          # .../src
PROJECT_DIR = SRC_DIR.parent                      # .../ (root progetto)
DATASETS_DIR = PROJECT_DIR / "datasets"


@dataclass(frozen=True)
class JobKBConfig:
    # Ora è indipendente da dove lanci python
    dataset_path: Path = DATASETS_DIR / "jobs-kb.csv"
    # Salviamo la KB dentro src/ per comodità (coerente con useKB.py)
    prolog_out: Path = SRC_DIR / "jobs_kb.pl"


class JobKnowledgeBaseBuilder:
    """
    Costruisce una KB Prolog con:
    - job/1
    - job_role/2, job_sector/2, job_seniority/2, job_contract/2, job_remote/2
    - job_salary/2, job_city/2
    - job_requires/2 (skill richieste)
    - regole: eligible/2, recommend_job/2
    """

    def __init__(self, cfg: JobKBConfig | None = None):
        self.cfg = cfg or JobKBConfig()

    def load(self) -> pd.DataFrame:
        if not self.cfg.dataset_path.exists():
            raise FileNotFoundError(
                "Dataset KB non trovato.\n"
                f"Sto cercando qui: {self.cfg.dataset_path.resolve()}\n\n"
                "Soluzione:\n"
                f"- crea il file jobs-kb.csv dentro: {DATASETS_DIR.resolve()}\n"
                "- oppure genera automaticamente jobs-kb.csv con lo script build_jobs_kb_from_supervised.py\n"
            )

        df = pd.read_csv(self.cfg.dataset_path)
        required_cols = {
            "job_id", "role", "sector", "seniority", "contract_type", "remote",
            "salary_offered", "location_city", "required_skills"
        }
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError("Mancano colonne nel jobs-kb.csv: " + ", ".join(missing))

        df = df.dropna(subset=["job_id", "role"]).copy()
        return df

    def write_prolog(self, df: pd.DataFrame) -> Path:
        out = self.cfg.prolog_out
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            f.write("% Knowledge Base - Matching Lavoro–Candidato\n\n")

            # --- FACTS ---
            f.write("% Fatti di base\n\n")

            seen_job = set()
            for _, row in df.iterrows():
                job_id = norm_atom(row["job_id"])
                if job_id in seen_job:
                    continue
                seen_job.add(job_id)

                role = norm_atom(row["role"])
                sector = norm_atom(row["sector"])
                seniority = norm_atom(row["seniority"])
                contract = norm_atom(row["contract_type"])
                remote = norm_atom(row["remote"])
                city = norm_atom(row["location_city"])

                salary = int(row["salary_offered"]) if str(row["salary_offered"]).strip() != "" else 0
                skills = split_skills(row["required_skills"])

                f.write(f"job({job_id}).\n")
                f.write(f"job_role({job_id}, {role}).\n")
                f.write(f"job_sector({job_id}, {sector}).\n")
                f.write(f"job_seniority({job_id}, {seniority}).\n")
                f.write(f"job_contract({job_id}, {contract}).\n")
                f.write(f"job_remote({job_id}, {remote}).\n")
                f.write(f"job_salary({job_id}, {salary}).\n")
                f.write(f"job_city({job_id}, {city}).\n")

                for sk in skills:
                    f.write(f"job_requires({job_id}, {sk}).\n")

                f.write("\n")

            # --- RULES ---
            f.write("% Regole (ragionamento)\n\n")

            f.write("""
% Un candidato è eleggibile per un job se:
%  - tutte le skill richieste dal job sono presenti nel profilo candidato
eligible(Cand, Job) :-
    job(Job),
    forall(job_requires(Job, S), candidato_skill(Cand, S)).

% Vincoli soft aggiuntivi (remote e salario):
salary_ok(Cand, Job) :-
    candidato_min_salary(Cand, MinS),
    job_salary(Job, Sal),
    Sal >= MinS.

remote_ok(Cand, Job) :-
    candidato_remote_ok(Cand, yes), !.
remote_ok(Cand, Job) :-
    candidato_remote_ok(Cand, no),
    job_remote(Job, no).

% Stessa città (se non remote)
city_ok(Cand, Job) :-
    job_remote(Job, yes), !.
city_ok(Cand, Job) :-
    candidato_city(Cand, C),
    job_city(Job, C).

% Raccomandazione "forte": eligible + vincoli
recommend_job(Cand, Job) :-
    eligible(Cand, Job),
    salary_ok(Cand, Job),
    remote_ok(Cand, Job),
    city_ok(Cand, Job).
""")

        return out

    def run(self):
        df = self.load()
        out = self.write_prolog(df)
        print("KB Prolog generata correttamente ✅")
        print("Dataset letto da:", self.cfg.dataset_path.resolve())
        print("KB salvata in:", out.resolve())


def main():
    builder = JobKnowledgeBaseBuilder()
    builder.run()


if __name__ == "__main__":
    main()