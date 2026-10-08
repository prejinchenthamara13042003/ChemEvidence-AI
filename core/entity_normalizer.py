"""
Chemical Entity Normalization Engine.
Standardizes chemical compound names, biological targets, IUPAC names,
molecular formulas, and harmonizes synonyms against standard reference databases.
"""

import re
from typing import List, Dict, Optional, Any
from core.models import CompoundInfo, NormalizedEntity, BioactivityResult
from core.pubchem_service import PubChemService


class EntityNormalizer:
    """Normalizes chemical and biological entities to standardized canonical representations."""

    KNOWN_DRUGS: Dict[str, Dict[str, Any]] = {
        "erlotinib": {
            "canonical_name": "Erlotinib",
            "entity_type": "Compound",
            "formula": "C22H23N3O4",
            "molecular_weight": 393.44,
            "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            "pubchem_cid": 28267319,
            "chembl_id": "CHEMBL547",
            "synonyms": ["Tarceva", "OSI-774", "CP-358774"]
        },
        "gefitinib": {
            "canonical_name": "Gefitinib",
            "entity_type": "Compound",
            "formula": "C22H24ClFN4O3",
            "molecular_weight": 446.9,
            "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "pubchem_cid": 123631,
            "chembl_id": "CHEMBL939",
            "synonyms": ["Iressa", "ZD1839"]
        },
        "staurosporine": {
            "canonical_name": "Staurosporine",
            "entity_type": "Compound",
            "formula": "C28H26N4O3",
            "molecular_weight": 466.5,
            "pubchem_cid": 44259,
            "chembl_id": "CHEMBL17684",
            "synonyms": ["AM-2282", "Antibiotic AM-2282"]
        },
        "osimertinib": {
            "canonical_name": "Osimertinib",
            "entity_type": "Compound",
            "formula": "C28H33N7O2",
            "molecular_weight": 499.6,
            "pubchem_cid": 71496458,
            "chembl_id": "CHEMBL3353410",
            "synonyms": ["Tagrisso", "AZD9291"]
        }
    }

    KNOWN_TARGETS: Dict[str, Dict[str, Any]] = {
        "egfr": {
            "canonical_name": "EGFR Kinase (Recombinant Human)",
            "entity_type": "Target",
            "synonyms": ["Epidermal Growth Factor Receptor", "ERBB1", "HER1", "wild-type EGFR", "EGFR WT"]
        },
        "cdk4": {
            "canonical_name": "CDK4 (Cyclin-Dependent Kinase 4)",
            "entity_type": "Target",
            "synonyms": ["Cyclin-dependent kinase 4", "CMM3", "PSK-J3"]
        },
        "a549": {
            "canonical_name": "A549 Human Lung Carcinoma Cell Line",
            "entity_type": "Target",
            "synonyms": ["A549 cells", "A549 adenocarcinoma"]
        },
        "hela": {
            "canonical_name": "HeLa Cervical Cancer Cell Line",
            "entity_type": "Target",
            "synonyms": ["HeLa cells"]
        }
    }

    @classmethod
    def normalize_compound(cls, comp: CompoundInfo) -> NormalizedEntity:
        """Standardizes a compound extracted from a paper."""
        raw_name = comp.name.strip()
        comp_id = comp.compound_id.strip()
        lower_name = raw_name.lower()
        lower_id = comp_id.lower()

        # Check known reference drugs
        for k, info in cls.KNOWN_DRUGS.items():
            if k in lower_name or k in lower_id:
                return NormalizedEntity(
                    original_text=f"{comp_id} ({raw_name})",
                    canonical_name=info["canonical_name"],
                    entity_type="Compound",
                    formula=info.get("formula") or comp.formula,
                    molecular_weight=info.get("molecular_weight") or comp.molecular_weight,
                    smiles=info.get("smiles") or comp.smiles,
                    pubchem_cid=info.get("pubchem_cid") or comp.pubchem_cid,
                    chembl_id=info.get("chembl_id"),
                    synonyms=[comp_id, raw_name] + info.get("synonyms", [])
                )

        # Standardize "Compound 3b", "Derivative 5c", etc.
        id_match = re.search(r"\b(Compound|Derivative|Inhibitor|Analogue)\s*([0-9]+[a-zA-Z]?)\b", f"{comp_id} {raw_name}", re.IGNORECASE)
        if id_match:
            kind = id_match.group(1).title()
            num = id_match.group(2).lower()
            canonical_label = f"{kind} {num}"
        else:
            canonical_label = raw_name.title()

        # Query PubChem for structure enrichment if not present
        formula = comp.formula
        mw = comp.molecular_weight
        smiles = comp.smiles
        cid = comp.pubchem_cid

        if not cid:
            pubchem_res = PubChemService.lookup_compound(raw_name) or PubChemService.lookup_compound(canonical_label)
            if pubchem_res:
                cid = pubchem_res.get("pubchem_cid")
                formula = formula or pubchem_res.get("formula")
                mw = mw or pubchem_res.get("molecular_weight")
                smiles = smiles or pubchem_res.get("smiles")

        synonyms = [comp_id, raw_name]
        if comp.chemical_class:
            synonyms.append(comp.chemical_class)

        return NormalizedEntity(
            original_text=f"{comp_id}: {raw_name}",
            canonical_name=canonical_label,
            entity_type="Compound",
            formula=formula,
            molecular_weight=mw,
            smiles=smiles,
            pubchem_cid=cid,
            synonyms=list(set(synonyms))
        )

    @classmethod
    def normalize_target(cls, target_name: str) -> NormalizedEntity:
        """Standardizes biological targets and cell lines."""
        raw = target_name.strip()
        lower_t = raw.lower()

        for k, info in cls.KNOWN_TARGETS.items():
            if k in lower_t or any(syn.lower() in lower_t for syn in info.get("synonyms", [])):
                return NormalizedEntity(
                    original_text=raw,
                    canonical_name=info["canonical_name"],
                    entity_type="Target",
                    synonyms=[raw] + info.get("synonyms", [])
                )

        return NormalizedEntity(
            original_text=raw,
            canonical_name=raw.title(),
            entity_type="Target",
            synonyms=[raw]
        )

    @classmethod
    def normalize_all(
        cls,
        compounds: List[CompoundInfo],
        bioactivities: List[BioactivityResult]
    ) -> List[NormalizedEntity]:
        """Normalizes all unique chemical entities and targets across the document."""
        normalized: List[NormalizedEntity] = []
        seen = set()

        for comp in compounds:
            norm = cls.normalize_compound(comp)
            if norm.canonical_name not in seen:
                seen.add(norm.canonical_name)
                normalized.append(norm)

        for b in bioactivities:
            if b.target:
                t_norm = cls.normalize_target(b.target)
                if t_norm.canonical_name not in seen:
                    seen.add(t_norm.canonical_name)
                    normalized.append(t_norm)

        return normalized
