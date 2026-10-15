"""
Chemistry-Specific Factual and Metric Accuracy Evaluator.
Evaluates entity recognition, target matching, numerical precision with unit conversion,
and strictly disambiguates biochemical metrics (IC50 vs GI50 vs EC50 vs Ki vs MIC).
"""

import re
import math
from typing import Dict, Any, Optional, Tuple, List
from evaluation.benchmark_loader import BenchmarkCase


# Normalized Metric Classifications
CANONICAL_METRICS = {
    "ic50": "IC50",
    "ic-50": "IC50",
    "ic₅₀": "IC50",
    "gi50": "GI50",
    "gi-50": "GI50",
    "gi₅₀": "GI50",
    "ec50": "EC50",
    "ec-50": "EC50",
    "ec₅₀": "EC50",
    "ki": "Ki",
    "kd": "Kd",
    "mic": "MIC",
    "lc50": "LC50",
    "ed50": "ED50",
    "cc50": "CC50",
    "tgi": "TGI",
    "inhibition": "% inhibition",
    "% inhibition": "% inhibition",
    "yield": "yield",
    "isolated yield": "yield",
    "purity": "purity"
}

# Metric Equivalence Groups (Metrics in different groups are strictly NOT equivalent)
METRIC_GROUPS = {
    "IC50": "enzymatic_inhibition",
    "Ki": "binding_affinity_ki",
    "Kd": "dissociation_constant_kd",
    "GI50": "cellular_growth_inhibition",
    "EC50": "effective_concentration",
    "CC50": "cytotoxicity_concentration",
    "MIC": "minimum_inhibitory_concentration",
    "LC50": "lethal_concentration",
    "ED50": "effective_dose",
    "TGI": "total_growth_inhibition",
    "% inhibition": "percent_inhibition",
    "yield": "synthetic_yield",
    "purity": "chemical_purity"
}


def normalize_metric_type(metric: Optional[str]) -> Optional[str]:
    """Normalizes metric string variants to canonical representations."""
    if not metric:
        return None
    m_clean = metric.lower().strip()
    m_clean = m_clean.replace("₅", "5").replace("₀", "0")
    m_clean = m_clean.replace("_", "").replace("{", "").replace("}", "")
    m_clean = re.sub(r"\\text", "", m_clean)
    m_clean = re.sub(r"\s+", " ", m_clean)
    return CANONICAL_METRICS.get(m_clean, metric.upper())


def parse_chemistry_entities_from_text(text: str) -> Dict[str, Any]:
    """
    Deterministically parses chemical compounds, biological targets,
    metric types, numerical measurements, and units from an answer string.
    """
    parsed: Dict[str, Any] = {
        "compounds": [],
        "targets": [],
        "metrics": [],
        "values": [],
        "units": []
    }
    if not text:
        return parsed

    # Strip LaTeX math tags and formatting (e.g. \text{IC}_{50}, \mathrm{...})
    text_clean = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text_clean = re.sub(r"\\(?:mathrm|mathbf|mathit)\{([^}]+)\}", r"\1", text_clean)
    text_clean = text_clean.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")
    text_clean = text_clean.replace("₅", "5").replace("₀", "0")
    text_clean = re.sub(r"[\u00b5\u03bc]", "µ", text_clean)
    text_clean = re.sub(r"\bic\s*_?50\b", "IC50", text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r"\bgi\s*_?50\b", "GI50", text_clean, flags=re.IGNORECASE)
    text_clean = re.sub(r"\bec\s*_?50\b", "EC50", text_clean, flags=re.IGNORECASE)

    # 1. Compound detection (e.g. Compound 3b, 5a, 5d, Erlotinib, etc.)
    comp_patterns = [
        r"\b(?:compound|derivative|analogue|analog|molecule|entry)\s+([0-9]+[a-z]?|[a-z][0-9]+)\b",
        r"\b([0-9]{1,3}[a-z])\b",
        r"\b(erlotinib|gefitinib|osimertinib|staurosporine|lapatinib|imatinib)\b"
    ]
    for pat in comp_patterns:
        for m in re.finditer(pat, text_clean, re.IGNORECASE):
            c_val = m.group(1).lower()
            if c_val not in parsed["compounds"]:
                parsed["compounds"].append(c_val)

    # 2. Metric detection (Strict extraction)
    # Check for specific metrics like IC50, GI50, EC50, Ki, Kd, MIC, yield
    metric_regex = r"\b(IC_?50|GI_?50|EC_?50|Ki|Kd|MIC|CC50|LC50|ED50|TGI|isolated\s+yield|yield|purity|%\s*inhibition)\b"
    for m in re.finditer(metric_regex, text_clean, re.IGNORECASE):
        norm_m = normalize_metric_type(m.group(1))
        if norm_m and norm_m not in parsed["metrics"]:
            parsed["metrics"].append(norm_m)

    # 3. Target detection (Enzymes, Receptors, Cell lines)
    target_patterns = [
        r"\b(EGFR|HER2|CDK4|CDK6|BRAF|VEGFR|ALK|ROS1|KRAS|mTOR|PI3K|PARP|HDAC|AChE|COX-2)\b",
        r"\b(A549|HeLa|HepG2|MCF-?7|Vero|PC-?9|H1975|HCT-?116|K562|Jurkat)\b"
    ]
    for pat in target_patterns:
        for m in re.finditer(pat, text_clean, re.IGNORECASE):
            tgt = m.group(1).upper().replace("-", "")
            if tgt not in parsed["targets"]:
                parsed["targets"].append(tgt)

    # 4. Values and Units extraction
    # Look for patterns like "8.4 nM", "8.4 ± 0.6 nM", "0.087 µM", "82%", "85 °C"
    val_unit_pattern = r"\b(\d+(?:\.\d+)?)\s*(?:[±\+\/\-]\s*\d+(?:\.\d+)?\s*)?(nM|µM|uM|mM|M|%|mol\s*%|°C|min|h|mg\/kg|µg\/mL|ug\/mL)\b"
    for m in re.finditer(val_unit_pattern, text_clean, re.IGNORECASE):
        val = float(m.group(1))
        unit = m.group(2).strip()
        parsed["values"].append(val)
        unit_norm = "µM" if unit in ["uM", "µM"] else unit
        if unit_norm not in parsed["units"]:
            parsed["units"].append(unit_norm)

    # Also capture isolated numbers if no unit directly attached
    if not parsed["values"]:
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text_clean):
            num = float(m.group(1))
            if num not in [50.0, 1.0, 2.0]:  # Skip common boilerplate numbers
                parsed["values"].append(num)

    return parsed


def convert_to_canonical_concentration(val: float, unit: str) -> Optional[float]:
    """
    Converts concentrations to canonical nanomolar (nM) units for comparison.
    1 M  = 1,000,000,000 nM
    1 mM = 1,000,000 nM
    1 µM = 1,000 nM
    1 nM = 1 nM
    """
    u = unit.lower().replace("μ", "µ")
    if u in ["nm", "nanomolar"]:
        return val
    elif u in ["µm", "um", "micromolar"]:
        return val * 1000.0
    elif u in ["mm", "millimolar"]:
        return val * 1000000.0
    elif u in ["m", "molar"]:
        return val * 1000000000.0
    return None


def are_numbers_equivalent(
    val_pred: float,
    unit_pred: Optional[str],
    val_expected: float,
    unit_expected: Optional[str],
    tolerance: float = 0.05
) -> bool:
    """
    Checks if predicted and expected numeric values are equivalent,
    including automatic unit conversions (e.g. 87 nM == 0.087 µM).
    """
    if val_pred is None or val_expected is None:
        return False

    # Check direct relative difference if units match or are missing
    if (not unit_pred and not unit_expected) or (unit_pred and unit_expected and unit_pred.lower() == unit_expected.lower()):
        diff = abs(val_pred - val_expected)
        denom = max(abs(val_expected), 1e-6)
        return (diff / denom) <= tolerance

    # Concentration unit conversion
    if unit_pred and unit_expected:
        canon_pred = convert_to_canonical_concentration(val_pred, unit_pred)
        canon_exp = convert_to_canonical_concentration(val_expected, unit_expected)
        if canon_pred is not None and canon_exp is not None:
            diff = abs(canon_pred - canon_exp)
            denom = max(abs(canon_exp), 1e-6)
            return (diff / denom) <= tolerance

    # If units weren't convertible, compare raw numbers
    diff = abs(val_pred - val_expected)
    denom = max(abs(val_expected), 1e-6)
    return (diff / denom) <= tolerance


def evaluate_chemistry_accuracy(
    predicted_text: str,
    case: BenchmarkCase
) -> Dict[str, Any]:
    """
    Evaluates fine-grained chemical factual accuracy:
    - Entity accuracy (Compound ID/name)
    - Target accuracy (Biological target/system)
    - Numeric accuracy (with unit conversions e.g. 87 nM == 0.087 µM)
    - Unit accuracy
    - Metric-type accuracy (STRICT distinction: IC50 != GI50 != EC50 != Ki)
    - Overall chemistry accuracy (all applicable components must match)
    """
    parsed = parse_chemistry_entities_from_text(predicted_text)

    # 1. Entity Accuracy (Compound)
    entity_correct = True
    if case.compound:
        exp_comp = case.compound.lower().strip()
        # Clean compound prefix like 'compound 3b' -> '3b'
        exp_comp_core = re.sub(r"^(?:compound|derivative|entry)\s+", "", exp_comp)
        
        # Check if the expected compound is present in extracted compounds
        found_comp = any(
            exp_comp_core == c or exp_comp_core in c or c in exp_comp_core
            for c in parsed["compounds"]
        )
        # Also check directly in raw predicted text
        if not found_comp:
            found_comp = bool(re.search(r"\b" + re.escape(exp_comp_core) + r"\b", predicted_text.lower()))

        entity_correct = found_comp

    # 2. Target Accuracy
    target_correct = True
    if case.target:
        exp_target = case.target.upper().replace("-", "").strip()
        found_target = any(
            exp_target in t or t in exp_target
            for t in parsed["targets"]
        )
        if not found_target:
            found_target = bool(re.search(r"\b" + re.escape(case.target.lower()) + r"\b", predicted_text.lower()))
        target_correct = found_target

    # 3. Metric-Type Accuracy (STRICT DISAMBIGUATION)
    metric_type_correct = True
    if case.metric:
        exp_metric_canonical = normalize_metric_type(case.metric)
        exp_group = METRIC_GROUPS.get(exp_metric_canonical, "other")

        # If prediction asserts a conflicting metric group (e.g. GI50 when IC50 was expected)
        # Check what metrics are in predicted
        pred_metrics = parsed["metrics"]

        if exp_metric_canonical:
            # Does the prediction contain the exact expected metric?
            has_exact = any(m == exp_metric_canonical for m in pred_metrics)
            if not has_exact:
                # Also check direct text
                has_exact = bool(re.search(r"\b" + re.escape(case.metric.lower()) + r"\b", predicted_text.lower()))

            # Did the prediction state a contradictory metric? (e.g. GI50 vs IC50)
            has_contradictory_metric = False
            for pm in pred_metrics:
                if METRIC_GROUPS.get(pm) != exp_group:
                    # e.g. GI50 vs IC50
                    if pm in ["GI50", "IC50", "EC50", "Ki", "Kd", "MIC", "yield"]:
                        has_contradictory_metric = True
                        break

            if has_contradictory_metric and not has_exact:
                metric_type_correct = False
            else:
                metric_type_correct = has_exact

    # 4. Numeric and Unit Accuracy
    numeric_correct = True
    unit_correct = True
    if case.value is not None:
        numeric_correct = False
        exp_unit = case.unit or ""

        # Test each numeric value parsed from the answer
        for p_val in parsed["values"]:
            # Check pairing with parsed units
            for p_unit in (parsed["units"] if parsed["units"] else [None]):
                if are_numbers_equivalent(p_val, p_unit, case.value, exp_unit):
                    numeric_correct = True
                    if exp_unit and p_unit:
                        # Recognized as equivalent unit or direct unit
                        unit_correct = True
                    break
            if numeric_correct:
                break

    # If case specifies unit but not value
    if case.unit and case.value is None:
        exp_u = case.unit.lower().replace("μ", "µ")
        unit_correct = any(exp_u in u.lower() for u in parsed["units"]) or (exp_u in predicted_text.lower())

    # Overall Chemistry Match (all specified criteria must be satisfied)
    conditions = []
    if case.compound:
        conditions.append(entity_correct)
    if case.target:
        conditions.append(target_correct)
    if case.metric:
        conditions.append(metric_type_correct)
    if case.value is not None:
        conditions.append(numeric_correct)

    overall_chemistry_correct = all(conditions) if conditions else entity_correct

    return {
        "entity_correct": entity_correct,
        "target_correct": target_correct,
        "metric_type_correct": metric_type_correct,
        "numeric_correct": numeric_correct,
        "unit_correct": unit_correct,
        "overall_chemistry_correct": overall_chemistry_correct,
        "parsed_entities": parsed
    }
