"""
PubChem REST API Service.
Enriches identified chemical compounds with 2D molecular structures,
IUPAC nomenclature, molecular weight, and canonical SMILES.
"""

import requests
from typing import Optional, Dict, Any


class PubChemService:
    """Provides chemical structure and property lookup via PubChem PUG REST API."""

    _CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    @classmethod
    def lookup_compound(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a chemical compound by name or SMILES.
        Returns dict with CID, formula, MW, IUPAC name, SMILES, and 2D image URL.
        """
        clean_query = query.strip()
        if not clean_query or len(clean_query) < 2:
            return None

        # Check in-memory cache
        cache_key = clean_query.lower()
        if cache_key in cls._CACHE:
            return cls._CACHE[cache_key]

        try:
            url = f"{cls.BASE_URL}/compound/name/{requests.utils.quote(clean_query)}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
            resp = requests.get(url, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                cid = props.get("CID")
                if cid:
                    result = {
                        "pubchem_cid": cid,
                        "formula": props.get("MolecularFormula"),
                        "molecular_weight": float(props.get("MolecularWeight", 0.0)),
                        "iupac_name": props.get("IUPACName"),
                        "smiles": props.get("CanonicalSMILES"),
                        "image_url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
                    }
                    cls._CACHE[cache_key] = result
                    return result
        except Exception:
            pass

        # If name fails, try lookup by SMILES if query looks like SMILES
        if any(c in clean_query for c in ["=", "#", "(", ")", "[", "]"]) and len(clean_query) > 4:
            try:
                smiles_url = f"{cls.BASE_URL}/compound/smiles/{requests.utils.quote(clean_query)}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
                resp = requests.get(smiles_url, timeout=3.5)
                if resp.status_code == 200:
                    data = resp.json()
                    props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                    cid = props.get("CID")
                    if cid:
                        result = {
                            "pubchem_cid": cid,
                            "formula": props.get("MolecularFormula"),
                            "molecular_weight": float(props.get("MolecularWeight", 0.0)),
                            "iupac_name": props.get("IUPACName"),
                            "smiles": props.get("CanonicalSMILES", clean_query),
                            "image_url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
                        }
                        cls._CACHE[cache_key] = result
                        return result
            except Exception:
                pass

        cls._CACHE[cache_key] = None
        return None
