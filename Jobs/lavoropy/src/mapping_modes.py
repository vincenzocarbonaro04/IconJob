"""
Mapping e vocabolario controllato
Dominio: Matching Lavoro–Candidato

Centralizza:
- normalizzazione skill
- mappatura ruoli → settori
- mappatura seniority → livello numerico
- cluster di skill (per ragionamento e analisi)
"""

from __future__ import annotations
import re


# -------------------------
# Skill normalization
# -------------------------

def normalize_skill(skill: str) -> str:
    """
    Normalizza una skill in formato standard.
    """
    if skill is None:
        return "unknown"

    s = skill.strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    s = re.sub(r"\s+", "_", s)

    # sinonimi comuni
    synonyms = {
        "py": "python",
        "python3": "python",
        "machinelearning": "ml",
        "machine_learning": "ml",
        "deep_learning": "dl",
        "javascript": "js",
        "nodejs": "node",
    }

    return synonyms.get(s, s)


# -------------------------
# Skill clusters (macro-aree tecniche)
# -------------------------

SKILL_CLUSTERS = {
    "data": {"python", "ml", "dl", "sql", "pandas", "numpy"},
    "web": {"js", "react", "node", "html", "css"},
    "devops": {"docker", "kubernetes", "aws", "azure"},
    "mobile": {"android", "ios", "swift", "kotlin"},
}


def skill_cluster(skill: str) -> str:
    s = normalize_skill(skill)
    for cluster, skills in SKILL_CLUSTERS.items():
        if s in skills:
            return cluster
    return "other"


# -------------------------
# Role → Sector mapping
# -------------------------

ROLE_TO_SECTOR = {
    "data_scientist": "tech",
    "machine_learning_engineer": "tech",
    "software_engineer": "tech",
    "backend_developer": "tech",
    "frontend_developer": "tech",
    "financial_analyst": "finance",
    "business_analyst": "business",
    "marketing_specialist": "marketing",
}


def role_to_sector(role: str) -> str:
    role = role.strip().lower().replace(" ", "_")
    return ROLE_TO_SECTOR.get(role, "other")


# -------------------------
# Seniority ordinal mapping
# -------------------------

SENIORITY_LEVEL = {
    "junior": 1,
    "mid": 2,
    "senior": 3,
}


def seniority_to_level(s: str) -> int:
    return SENIORITY_LEVEL.get(s.strip().lower(), 0)


# -------------------------
# Contract priority (soft preference)
# -------------------------

CONTRACT_PRIORITY = {
    "internship": 1,
    "part_time": 2,
    "full_time": 3,
    "freelance": 2,
}


def contract_priority(contract: str) -> int:
    return CONTRACT_PRIORITY.get(contract.strip().lower(), 0)