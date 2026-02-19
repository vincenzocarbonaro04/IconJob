"""
Rappresentazione e ragionamento - Uso Knowledge Base (Python-only)
Dominio: Matching Lavoro–Candidato

Questa versione NON usa pyswip/SWI-Prolog.
Carica datasets/jobs-kb.csv e applica regole rule-based equivalenti a:

- eligible(Cand, Job): tutte le skill richieste dal job sono presenti nel candidato
- recommend_job(Cand, Job): eligible + vincoli (salario, remote, città)

Dataset atteso: datasets/jobs-kb.csv
Colonne:
job_id, role, sector, seniority, contract_type, remote, salary_offered, location_city, required_skills
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


# -----------------------
# PATH ROBUSTI
# -----------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATASETS_DIR = PROJECT_DIR / "datasets"


def norm_atom(s: str) -> str:
    """Normalizza stringhe (skill/città/ruoli) in modo coerente e comparabile."""
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("_")
    return s if s else "unknown"


def parse_skills(raw: str) -> set[str]:
    """Parsa una lista skill 'python,ml,sql' in set normalizzato."""
    if raw is None:
        return set()
    txt = str(raw).strip()
    if not txt:
        return set()
    parts = [norm_atom(x) for x in txt.split(",")]
    return {p for p in parts if p and p != "unknown"}


@dataclass
class KBConfig:
    kb_csv_path: Path = DATASETS_DIR / "jobs-kb.csv"


class JobKBReasoner:
    def __init__(self, cfg: KBConfig | None = None):
        self.cfg = cfg or KBConfig()
        self.df: pd.DataFrame | None = None

    def load_kb(self) -> pd.DataFrame:
        if not self.cfg.kb_csv_path.exists():
            raise FileNotFoundError(
                "KB CSV non trovata.\n"
                f"Sto cercando qui: {self.cfg.kb_csv_path.resolve()}\n\n"
                "Soluzione: crea datasets/jobs-kb.csv (quello che abbiamo preparato)."
            )

        df = pd.read_csv(self.cfg.kb_csv_path)

        required_cols = {
            "job_id", "role", "sector", "seniority", "contract_type", "remote",
            "salary_offered", "location_city", "required_skills"
        }
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError("Mancano colonne nel jobs-kb.csv: " + ", ".join(missing))

        # normalizza campi base
        df = df.dropna(subset=["job_id", "role"]).copy()
        df["job_id"] = df["job_id"].astype(str).map(norm_atom)
        df["role"] = df["role"].astype(str).map(norm_atom)
        df["sector"] = df["sector"].astype(str).map(norm_atom)
        df["seniority"] = df["seniority"].astype(str).map(norm_atom)
        df["contract_type"] = df["contract_type"].astype(str).map(norm_atom)
        df["remote"] = df["remote"].astype(str).map(norm_atom)
        df["location_city"] = df["location_city"].astype(str).map(norm_atom)
        df["salary_offered"] = df["salary_offered"].astype(int)

        # skill come set
        df["required_skills_set"] = df["required_skills"].apply(parse_skills)

        self.df = df
        return df

    # --------- RULES (equivalenti a Prolog) ---------

    @staticmethod
    def eligible(candidate_skills: set[str], job_required_skills: set[str]) -> bool:
        """Tutte le skill richieste dal job sono contenute nel set del candidato."""
        return job_required_skills.issubset(candidate_skills)

    @staticmethod
    def salary_ok(min_salary: int, job_salary: int) -> bool:
        return job_salary >= min_salary

    @staticmethod
    def remote_ok(candidate_remote_ok: bool, job_remote: str) -> bool:
        job_remote = norm_atom(job_remote)
        if candidate_remote_ok:
            return True
        # se il candidato NON accetta remote, allora il job deve essere no-remote
        return job_remote == "no"

    @staticmethod
    def city_ok(candidate_city: str, job_city: str, job_remote: str) -> bool:
        # se il job è remote, la città non vincola
        if norm_atom(job_remote) == "yes":
            return True
        return norm_atom(candidate_city) == norm_atom(job_city)

    def query_eligible(self, cand_skills: set[str], limit: int = 10) -> pd.DataFrame:
        assert self.df is not None
        df = self.df.copy()
        df["is_eligible"] = df["required_skills_set"].apply(lambda req: self.eligible(cand_skills, req))
        res = df[df["is_eligible"]].copy()
        return res.head(limit)

    def query_recommend(
        self,
        cand_skills: set[str],
        cand_city: str,
        min_salary: int,
        cand_remote_ok: bool,
        limit: int = 10,
    ) -> pd.DataFrame:
        assert self.df is not None
        df = self.df.copy()

        def rec(row) -> bool:
            return (
                self.eligible(cand_skills, row["required_skills_set"])
                and self.salary_ok(min_salary, int(row["salary_offered"]))
                and self.remote_ok(cand_remote_ok, row["remote"])
                and self.city_ok(cand_city, row["location_city"], row["remote"])
            )

        df["is_recommended"] = df.apply(rec, axis=1)
        res = df[df["is_recommended"]].copy()

        # ranking semplice: prima salari alti, poi remote yes, poi meno skill richieste
        res["n_required_skills"] = res["required_skills_set"].apply(len)
        res = res.sort_values(by=["salary_offered", "remote", "n_required_skills"], ascending=[False, False, True])

        return res.head(limit)

    # --------- Utils ---------

    def list_roles(self, limit: int = 50) -> list[str]:
        assert self.df is not None
        roles = sorted(self.df["role"].unique().tolist())
        return roles[:limit]

    def list_skills(self, limit: int = 80) -> list[str]:
        assert self.df is not None
        all_sk = set()
        for s in self.df["required_skills_set"]:
            all_sk |= set(s)
        skills = sorted(all_sk)
        return skills[:limit]


def main():
    kb = JobKBReasoner()
    kb.load_kb()

    print("\n=== Query KB (Python-only) - Matching Lavoro–Candidato ===")
    print("KB caricata da:", kb.cfg.kb_csv_path.resolve())

    cand_id = input("\nID candidato (es. cand_1): ").strip() or "cand_1"
    skills_raw = input("Skill candidato (comma-separated, es: python,sql,ml): ").strip()
    cand_skills = parse_skills(skills_raw)

    cand_city = input("Città candidato (es. milano): ").strip() or "milano"
    min_salary = input("Salario minimo (es. 35000): ").strip() or "35000"
    remote_ok = input("Remote ok? (yes/no): ").strip().lower() or "yes"
    cand_remote_ok = (remote_ok == "yes")

    print(f"\nCandidato: {cand_id}")
    print("Skills:", sorted(list(cand_skills)))
    print("Città:", norm_atom(cand_city))
    print("Min salary:", int(min_salary))
    print("Remote ok:", cand_remote_ok)

    while True:
        print("\nMenu:")
        print("1) Lista ruoli disponibili (parziale)")
        print("2) Lista skill disponibili (parziale)")
        print("3) Query eligible (solo skill)")
        print("4) Query recommend (skill + vincoli)")
        print("0) Esci")

        choice = input("Scelta: ").strip()

        if choice == "1":
            roles = kb.list_roles()
            print("\nRuoli:")
            for r in roles:
                print("-", r)

        elif choice == "2":
            sk = kb.list_skills()
            print("\nSkill:")
            for s in sk:
                print("-", s)

        elif choice == "3":
            res = kb.query_eligible(cand_skills, limit=15)
            print("\nJob eleggibili (prime 15):")
            if res.empty:
                print("Nessun job eleggibile trovato.")
            else:
                print(res[["job_id", "role", "sector", "seniority", "contract_type", "remote", "salary_offered", "location_city", "required_skills"]].to_string(index=False))

        elif choice == "4":
            res = kb.query_recommend(
                cand_skills=cand_skills,
                cand_city=cand_city,
                min_salary=int(min_salary),
                cand_remote_ok=cand_remote_ok,
                limit=15,
            )
            print("\nJob raccomandati (prime 15):")
            if res.empty:
                print("Nessun job raccomandato trovato.")
            else:
                print(res[["job_id", "role", "sector", "seniority", "contract_type", "remote", "salary_offered", "location_city", "required_skills"]].to_string(index=False))

        elif choice == "0":
            break
        else:
            print("Scelta non valida.")


if __name__ == "__main__":
    main()