"""
Chemical Formula and Molecular Weight Calculation Engine.
Calculates exact molecular weights using IUPAC standard atomic weights,
extracts formulas from tables and HRMS / mass spectrometry data,
and links formulas and weights document-wide to identified compounds.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from core.pdf_parser import PageContent
from core.models import CompoundInfo, ExtractedTable


class ChemCalculator:
    """Calculates molecular properties and binds formulas and molecular weights to chemical entities."""

    # Standard IUPAC atomic weights (g/mol)
    ATOMIC_WEIGHTS: Dict[str, float] = {
        "H": 1.008,
        "He": 4.0026,
        "Li": 6.94,
        "Be": 9.0122,
        "B": 10.81,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Ne": 20.180,
        "Na": 22.990,
        "Mg": 24.305,
        "Al": 26.982,
        "Si": 28.085,
        "P": 30.974,
        "S": 32.06,
        "Cl": 35.45,
        "Ar": 39.948,
        "K": 39.098,
        "Ca": 40.078,
        "Sc": 44.956,
        "Ti": 47.867,
        "V": 50.942,
        "Cr": 51.996,
        "Mn": 54.938,
        "Fe": 55.845,
        "Co": 58.933,
        "Ni": 58.693,
        "Cu": 63.546,
        "Zn": 65.38,
        "Ga": 69.723,
        "Ge": 72.630,
        "As": 74.922,
        "Se": 78.971,
        "Br": 79.904,
        "Kr": 83.798,
        "Rb": 85.468,
        "Sr": 87.62,
        "Y": 88.906,
        "Zr": 91.224,
        "Nb": 92.906,
        "Mo": 95.95,
        "Ru": 101.07,
        "Rh": 102.91,
        "Pd": 106.42,
        "Ag": 107.87,
        "Cd": 112.41,
        "In": 114.82,
        "Sn": 118.71,
        "Sb": 121.76,
        "Te": 127.60,
        "I": 126.904,
        "Ba": 137.33,
        "Pt": 195.08,
        "Au": 196.97,
        "Hg": 200.59
    }

    # Unicode subscript character mapping
    SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    @classmethod
    def normalize_formula(cls, formula_raw: str) -> str:
        """Normalizes unicode subscripts, removes charges/brackets, and strips whitespace."""
        if not formula_raw:
            return ""
        norm = formula_raw.translate(cls.SUBSCRIPT_MAP).strip()
        norm = re.sub(r"\s+", "", norm)
        norm = re.sub(r"\[.*?\]\+?", "", norm)
        norm = norm.replace("+", "").replace("-", "")
        return norm

    @classmethod
    def calculate_molecular_weight(cls, formula: str) -> Optional[float]:
        """Calculates molecular weight from a Hill-system or standard molecular formula."""
        if not formula or not isinstance(formula, str):
            return None

        clean_formula = cls.normalize_formula(formula)

        # Must start with an element symbol
        if not re.match(r"^[A-Z]", clean_formula):
            return None

        # Parse element counts (e.g. C19H22N4O2)
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", clean_formula)
        if not matches:
            return None

        total_mw = 0.0
        parsed_chars = 0

        for elem, count_str in matches:
            if elem not in cls.ATOMIC_WEIGHTS:
                return None
            count = int(count_str) if count_str else 1
            total_mw += cls.ATOMIC_WEIGHTS[elem] * count
            parsed_chars += len(elem) + len(count_str)

        # Sanity check: formula should have parsed cleanly
        if parsed_chars < len(clean_formula) * 0.75:
            return None

        return round(total_mw, 2) if total_mw > 0 else None

    @classmethod
    def neutral_formula_from_ion(cls, ion_formula: str) -> str:
        """Converts an [M+H]+ or ionized formula to neutral formula by subtracting 1 H."""
        norm = cls.normalize_formula(ion_formula)
        h_match = re.search(r"H(\d+)", norm)
        if h_match:
            h_count = int(h_match.group(1))
            if h_count > 1:
                return re.sub(r"H\d+", f"H{h_count - 1}", norm, count=1)
        return norm

    @classmethod
    def bind_formulas_and_weights_document_wide(
        cls,
        compounds: List[CompoundInfo],
        tables: List[ExtractedTable],
        pages: List[PageContent]
    ) -> None:
        """
        Cross-references all document tables, experimental characterization blocks,
        and HRMS/MS spectra to populate missing molecular formulas and molecular weights.
        """
        full_text = "\n".join([p.text for p in pages])
        full_text_normalized = full_text.translate(cls.SUBSCRIPT_MAP)

        # 1. First, check all Extracted Tables for Compound <-> Formula / MW rows
        table_mappings: Dict[str, Dict[str, Any]] = {}
        for t in tables:
            headers_lower = [h.lower().strip() for h in t.headers]
            comp_col_idx = -1
            formula_col_idx = -1
            mw_col_idx = -1

            for idx, h in enumerate(headers_lower):
                if any(k in h for k in ["compound", "compd", "cpd", "derivative", "analogue", "inhibitor", "entry", "no.", "molecule"]):
                    comp_col_idx = idx
                elif any(k in h for k in ["formula", "mol. formula", "molecular formula", "m.f.", "mf", "chem. formula"]):
                    formula_col_idx = idx
                elif any(k in h for k in ["mw", "m.w.", "mol. wt", "mol wt", "molecular weight", "exact mass", "mass", "weight"]):
                    mw_col_idx = idx

            if comp_col_idx != -1 and (formula_col_idx != -1 or mw_col_idx != -1):
                for row in t.rows:
                    if len(row) > comp_col_idx:
                        c_val = row[comp_col_idx].strip()
                        c_key = re.sub(r"[^a-zA-Z0-9]", "", c_val).lower()
                        short_key = re.sub(r"^(compound|derivative|cpd|entry|no)", "", c_key).strip()
                        keys_to_register = [k for k in [c_key, short_key] if k]

                        for k_reg in keys_to_register:
                            table_mappings.setdefault(k_reg, {})
                            if formula_col_idx != -1 and len(row) > formula_col_idx:
                                form_val = cls.normalize_formula(row[formula_col_idx])
                                if re.match(r"^C\d+H\d+[A-Za-z0-9]*$", form_val):
                                    table_mappings[k_reg]["formula"] = form_val
                            if mw_col_idx != -1 and len(row) > mw_col_idx:
                                mw_val = row[mw_col_idx].strip()
                                num_m = re.search(r"(\d+(?:\.\d+)?)", mw_val)
                                if num_m:
                                    table_mappings[k_reg]["mw"] = float(num_m.group(1))

        # 2. Iterate through compounds and populate
        for comp in compounds:
            c_id = comp.compound_id.strip()
            c_key = re.sub(r"[^a-zA-Z0-9]", "", c_id).lower()
            short_id = re.sub(r"^(compound|derivative|inhibitor|cpd)", "", c_key).strip()
            label_num = re.search(r"\d+[a-zA-Z]?", c_id)
            num_only = label_num.group(0).lower() if label_num else ""

            # Check table mappings
            matched_table_data = (
                table_mappings.get(c_key)
                or table_mappings.get(short_id)
                or (table_mappings.get(num_only) if num_only else None)
            )
            if matched_table_data:
                if not comp.formula and "formula" in matched_table_data:
                    comp.formula = matched_table_data["formula"]
                if not comp.molecular_weight and "mw" in matched_table_data:
                    comp.molecular_weight = matched_table_data["mw"]

            # If formula is still missing, scan document characterization blocks
            if not comp.formula:
                regex_compound = re.escape(c_id)
                label_pattern = f"(?:{regex_compound}|\\b{label_num.group(0)}\\b)" if label_num else regex_compound

                # Search within text windows of 800 characters around the compound label
                for match in re.finditer(label_pattern, full_text_normalized, re.IGNORECASE):
                    start = max(0, match.start() - 100)
                    end = min(len(full_text_normalized), match.end() + 800)
                    window = full_text_normalized[start:end]

                    # Pattern A: HRMS calculated for / calcd for
                    hrms_match = re.search(
                        r"(?:calcd|calculated)\s+(?:for\s+)?(C\d+H\d+[A-Za-z0-9]*)\s*(?:\[M\s*\+\s*H\]\+)?",
                        window,
                        re.IGNORECASE
                    )
                    if hrms_match:
                        raw_ion = hrms_match.group(1)
                        if "[m+h]+" in window[hrms_match.end():hrms_match.end()+20].lower():
                            comp.formula = cls.neutral_formula_from_ion(raw_ion)
                        else:
                            comp.formula = cls.normalize_formula(raw_ion)
                        break

                    # Pattern B: Elemental analysis (Anal. Calcd for C24H27N5O3)
                    anal_match = re.search(
                        r"Anal\.\s*(?:Calcd|calcd)?\s*(?:for)?\s*[:\s(]*(C\d+H\d+[A-Za-z0-9]*)",
                        window,
                        re.IGNORECASE
                    )
                    if anal_match:
                        comp.formula = cls.normalize_formula(anal_match.group(1))
                        break

                    # Pattern C: Parenthetical formula, e.g. Compound 7j (C24H27N5O3)
                    paren_match = re.search(r"\(\s*(C\d+H\d+[A-Za-z0-9]*)\s*\)", window)
                    if paren_match:
                        comp.formula = cls.normalize_formula(paren_match.group(1))
                        break

                    # Pattern D: Explicit formula mention: formula C24H27N5O3 or formula: C24H27N5O3
                    form_kw_match = re.search(r"(?:formula|MF)\s*[:=]?\s*(C\d+H\d+[A-Za-z0-9]*)", window, re.IGNORECASE)
                    if form_kw_match:
                        comp.formula = cls.normalize_formula(form_kw_match.group(1))
                        break

                    # Pattern E: Spaced formula from PDF extractors (e.g. C 19 H 22 N 4 O 2)
                    spaced_match = re.search(r"\b(C\s*\d{1,3}\s*H\s*\d{1,3}(?:\s*[A-Z][a-z]?\s*\d*)*)\b", window)
                    if spaced_match:
                        cand = cls.normalize_formula(spaced_match.group(1))
                        if re.match(r"^C\d+H\d+[A-Za-z0-9]*$", cand) and not any(cand.startswith(x) for x in ["CDCl", "DMSO", "EtOAc"]):
                            comp.formula = cand
                            break

            # 3. If formula is found, compute molecular weight immediately using atomic weights
            if comp.formula and not comp.molecular_weight:
                calc_val = cls.calculate_molecular_weight(comp.formula)
                if calc_val:
                    comp.molecular_weight = calc_val

            # 4. If molecular weight is still missing, scan for HRMS / MS values
            if not comp.molecular_weight:
                regex_compound = re.escape(c_id)
                label_pattern = f"(?:{regex_compound}|\\b{label_num.group(0)}\\b)" if label_num else regex_compound
                for match in re.finditer(label_pattern, full_text_normalized, re.IGNORECASE):
                    window = full_text_normalized[max(0, match.start() - 50):min(len(full_text_normalized), match.end() + 600)]
                    ms_match = re.search(
                        r"m/z\s*(?:calcd|calculated|found)?\s*(?:for\s+[A-Za-z0-9]+\s*)?(?:\[M\s*\+\s*H\]\+)?\s*[:=]?\s*(\d{2,4}\.\d{1,4})",
                        window,
                        re.IGNORECASE
                    )
                    if ms_match:
                        ion_val = float(ms_match.group(1))
                        # If [M+H]+, subtract 1.008 to get neutral MW
                        comp.molecular_weight = round(ion_val - 1.008, 1)
                        break

                    # Explicit MW pattern: MW 433.5 or Mol. Wt. 433.5
                    mw_explicit = re.search(r"\b(?:MW|mol\.\s*wt\.?|molecular\s+weight)\s*[:=]?\s*(\d{2,4}(?:\.\d{1,2})?)\b", window, re.IGNORECASE)
                    if mw_explicit:
                        comp.molecular_weight = round(float(mw_explicit.group(1)), 1)
                        break

            # 5. Extract specific chemical IUPAC name if available in characterization block
            if comp.name.startswith("Quinazoline Compound") or comp.name.startswith("Pyrimidine Compound") or comp.name.startswith("Heterocycle Compound") or "Compound" in comp.name:
                regex_compound = re.escape(c_id)
                iupac_match = re.search(r"([A-Z][a-zA-Z0-9\-\,\(\)\[\]]{8,80})\s*\(\s*(?:" + regex_compound + r"|" + (re.escape(num_only) if num_only else "___") + r")\s*\)", full_text_normalized, re.IGNORECASE)
                if iupac_match:
                    cand_name = iupac_match.group(1).strip()
                    if not any(k in cand_name.lower() for k in ["table", "scheme", "figure", "equation"]):
                        comp.name = cand_name

