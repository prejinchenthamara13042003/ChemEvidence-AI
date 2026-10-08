"""
Multilingual Chemistry Assistant Engine (ChemBot).
Provides proper, simple, accessible chemistry explanations in the user's native language
(Malayalam, Hindi, Tamil, German, Spanish, French, Chinese, Arabic, English, etc.)
with structured sections: Simple Answer, Key Chemical Facts, Everyday Meaning, and Grounded Citations.
Robust to API rate limits with an intelligent local multilingual reasoning fallback.
"""

import os
import re
import html
from typing import List, Dict, Any, Optional, Tuple
from core.models import PaperAnalysisResult, EvidenceCitation, ExperimentalCondition, MultilingualBotResponse
from core.pdf_parser import PageContent

# Gemini GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class MultilingualChemBot:
    """
    Multilingual Chemistry Assistant.
    Detects user language, synthesizes plain-language chemical explanations,
    and supports multi-turn conversational queries both for loaded papers and general chemistry.
    """

    SUPPORTED_LANGUAGES = {
        "en": {"name": "English", "native": "English", "flag": "🇬🇧"},
        "ml": {"name": "Malayalam", "native": "മലയാളം", "flag": "🇮🇳"},
        "hi": {"name": "Hindi", "native": "हिंदी", "flag": "🇮🇳"},
        "ta": {"name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳"},
        "de": {"name": "German", "native": "Deutsch", "flag": "🇩🇪"},
        "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸"},
        "fr": {"name": "French", "native": "Français", "flag": "🇫🇷"},
        "ar": {"name": "Arabic", "native": "العربية", "flag": "🇸🇦"},
        "zh": {"name": "Chinese", "native": "中文", "flag": "🇨🇳"},
    }

    LOCALIZED_HEADERS = {
        "ml": {
            "simple_title": "🎯 ലളിതമായ ഉത്തരം (Simple Answer)",
            "facts_title": "🔬 പ്രധാന രാസവിവരങ്ങളും സംഖ്യകളും (Key Chemical Facts)",
            "meaning_title": "💡 ഇത് എന്തുകൊണ്ട് പ്രധാനം? / ലളിതമായ വിശദീകരണം (Why It Matters)",
            "source_title": "📚 രേഖാമൂലമുള്ള തെളിവ് (Evidence & Source)",
            "general_note": "പൊതുവായ രസതന്ത്ര തത്വം (General Chemistry Knowledge)",
            "not_found_note": "ഈ ഗവേഷണ പേപ്പറിൽ ഇത് വ്യക്തമായി പരാമർശിച്ചിട്ടില്ല, എന്നാൽ പൊതു രസതന്ത്രം പ്രകാരം:",
        },
        "hi": {
            "simple_title": "🎯 सरल उत्तर (Simple Answer)",
            "facts_title": "🔬 मुख्य रासायनिक तथ्य और आंकड़े (Key Facts & Numbers)",
            "meaning_title": "💡 यह क्यों महत्वपूर्ण है? / आसान शब्दों में (Why It Matters)",
            "source_title": "📚 शोध पत्र का प्रमाण और स्रोत (Evidence & Source)",
            "general_note": "सामान्य रसायन विज्ञान सिद्धांत (General Chemistry Knowledge)",
            "not_found_note": "इस शोध पत्र में इसका सीधा उल्लेख नहीं है, लेकिन सामान्य रसायन विज्ञान के अनुसार:",
        },
        "ta": {
            "simple_title": "🎯 எளிய பதில் (Simple Answer)",
            "facts_title": "🔬 முக்கிய வேதியியல் தகவல்கள் மற்றும் எண்கள் (Key Facts)",
            "meaning_title": "💡 இது ஏன் முக்கியம்? / எளிய விளக்கம் (Why It Matters)",
            "source_title": "📚 ஆய்வுக் கட்டுரை ஆதாரம் (Evidence & Source)",
            "general_note": "பொது வேதியியல் அறிவு (General Chemistry Knowledge)",
            "not_found_note": "இந்த ஆய்வுக் கட்டுரையில் நேரடியாகக் குறிப்பிடப்படவில்லை, ஆனால் பொது வேதியியல் கோட்பாட்டின்படி:",
        },
        "de": {
            "simple_title": "🎯 Einfache und direkte Antwort (Simple Answer)",
            "facts_title": "🔬 Wichtige chemische Fakten & Kennzahlen (Key Facts)",
            "meaning_title": "💡 Warum das wichtig ist / Im Klartext (Why It Matters)",
            "source_title": "📚 Wissenschaftlicher Beleg & Quelle (Evidence & Source)",
            "general_note": "Allgemeines chemisches Fachwissen (General Chemistry Knowledge)",
            "not_found_note": "In diesem Manuskript nicht direkt erwähnt, aber nach allgemeiner Fachliteratur:",
        },
        "es": {
            "simple_title": "🎯 Respuesta Simple y Directa (Simple Answer)",
            "facts_title": "🔬 Datos y Cifras Químicas Clave (Key Facts & Numbers)",
            "meaning_title": "💡 ¿Por qué es importante? / En palabras sencillas (Why It Matters)",
            "source_title": "📚 Evidencia y Fuente del Artículo (Evidence & Source)",
            "general_note": "Conocimiento químico general (General Chemistry Knowledge)",
            "not_found_note": "No se menciona explícitamente en este manuscrito, pero según la química general:",
        },
        "fr": {
            "simple_title": "🎯 Réponse Simple et Claire (Simple Answer)",
            "facts_title": "🔬 Faits et Données Chimiques Majeurs (Key Facts & Numbers)",
            "meaning_title": "💡 Pourquoi c'est important / En termes simples (Why It Matters)",
            "source_title": "📚 Preuves et Source du Document (Evidence & Source)",
            "general_note": "Connaissances chimiques générales (General Chemistry Knowledge)",
            "not_found_note": "Non mentionné directement dans ce document, mais selon les principes de chimie :",
        },
        "zh": {
            "simple_title": "🎯 简明直观解答 (Simple Answer)",
            "facts_title": "🔬 核心化学事实与关键数据 (Key Facts & Numbers)",
            "meaning_title": "💡 通俗解读 / 为什么重要 (Why It Matters)",
            "source_title": "📚 论文依据与出处引用 (Evidence & Source)",
            "general_note": "化学通用常识 (General Chemistry Knowledge)",
            "not_found_note": "本篇论文未直接记载该数据，但依据通用化学原理：",
        },
        "ar": {
            "simple_title": "🎯 إجابة مبسطة ومباشرة (Simple Answer)",
            "facts_title": "🔬 حقائق وأرقام كيميائية رئيسية (Key Facts & Numbers)",
            "meaning_title": "💡 لماذا هذا مهم؟ / بكلمات بسيطة (Why It Matters)",
            "source_title": "📚 أدلة ومصدر البحث (Evidence & Source)",
            "general_note": "معرفة كيميائية عامة (General Chemistry Knowledge)",
            "not_found_note": "لم يرد هذا صراحة في هذا البحث، ولكن وفقاً للمبادئ الكيميائية العامة:",
        },
        "en": {
            "simple_title": "🎯 Simple & Direct Answer",
            "facts_title": "🔬 Key Chemical Facts & Numbers",
            "meaning_title": "💡 Why It Matters / In Plain Words",
            "source_title": "📚 Evidence & Paper Source",
            "general_note": "Established Chemistry Benchmark (General Chemistry Knowledge)",
            "not_found_note": "Not explicitly detailed in this publication, but according to broader chemical literature:",
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=self.api_key, http_options={"timeout": 15000})
            except Exception as e:
                print(f"[MultilingualChemBot] Client init error: {e}")
                self.client = None

    @classmethod
    def detect_language(cls, text: str, user_choice: str = "Auto-Detect") -> Tuple[str, str]:
        """
        Detects the language of the prompt or honors user selection.
        Returns (language_code, language_display_name).
        """
        # 1. User manual override
        if user_choice and user_choice != "Auto-Detect":
            clean_choice = user_choice.lower()
            for code, meta in cls.SUPPORTED_LANGUAGES.items():
                if (code in clean_choice or 
                    meta["name"].lower() in clean_choice or 
                    meta["native"].lower() in clean_choice):
                    return code, meta["name"]

        # 2. Unicode script-based automatic detection
        if re.search(r"[\u0D00-\u0D7F]", text):
            return "ml", "Malayalam"
        if re.search(r"[\u0900-\u097F]", text):
            return "hi", "Hindi"
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "ta", "Tamil"
        if re.search(r"[\u0600-\u06FF]", text):
            return "ar", "Arabic"
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh", "Chinese"

        # 3. European Latin-based linguistic patterns
        lower_t = text.lower()

        # French: distinctive French words or accents
        if (re.search(r"\b(quel|quelle|quels|quelles|comment|pourquoi|cette|dans|avec|lequel|laquelle|rendement|est-ce)\b", lower_t) or
            re.search(r"[çàùâêîôûœ]", lower_t)):
            return "fr", "French"

        # German: umlauts or common question/chemistry vocabulary
        if (re.search(r"[äöüß]", lower_t) or
            re.search(r"\b(warum|welche|welcher|welches|ausbeute|hemmstoff|reaktion|verbindung|ist das|wie funktioniert|der|die|das)\b", lower_t)):
            return "de", "German"

        # Spanish: inverted punctuation, tildes (ñ), or common terms
        if (re.search(r"[¿¡ñ]", lower_t) or
            re.search(r"\b(cuál|cuáles|cómo|por qué|compuesto|rendimiento|inhibidor|reacción|qué es)\b", lower_t)):
            return "es", "Spanish"

        # Default fallback
        return "en", "English"

    def answer(
        self,
        query: str,
        analysis: Optional[PaperAnalysisResult] = None,
        pages: Optional[List[PageContent]] = None,
        language_choice: str = "Auto-Detect",
        simplicity_level: str = "Simple & Everyday"
    ) -> MultilingualBotResponse:
        """
        Main entry point for generating a proper simple answer in the user's language.
        Tries Gemini 3.8 Flash first; automatically falls back to smart local synthesis.
        """
        lang_code, lang_name = self.detect_language(query, language_choice)
        headers = self.LOCALIZED_HEADERS.get(lang_code, self.LOCALIZED_HEADERS["en"])

        # Attempt Gemini 3.8 Flash first if available
        if self.client and HAS_GENAI:
            try:
                gemini_resp = self._call_gemini(
                    query=query,
                    lang_code=lang_code,
                    lang_name=lang_name,
                    analysis=analysis,
                    pages=pages,
                    simplicity_level=simplicity_level
                )
                if gemini_resp:
                    return gemini_resp
            except Exception as e:
                print(f"[MultilingualChemBot] Gemini API call skipped/failed ({e}). Switching to local engine.")

        # Local Multilingual Reasoning Fallback
        return self._local_multilingual_synthesis(
            query=query,
            lang_code=lang_code,
            lang_name=lang_name,
            headers=headers,
            analysis=analysis,
            pages=pages,
            simplicity_level=simplicity_level
        )

    def _call_gemini(
        self,
        query: str,
        lang_code: str,
        lang_name: str,
        analysis: Optional[PaperAnalysisResult],
        pages: Optional[List[PageContent]],
        simplicity_level: str
    ) -> Optional[MultilingualBotResponse]:
        """Queries Gemini 3.8 Flash with strict instructions to reply in the requested language."""
        headers = self.LOCALIZED_HEADERS.get(lang_code, self.LOCALIZED_HEADERS["en"])
        native_name = self.SUPPORTED_LANGUAGES.get(lang_code, {}).get("native", lang_name)

        # Prepare context excerpt from loaded paper
        paper_context_str = "No specific manuscript uploaded. Answer as a general, friendly multilingual chemistry mentor."
        if analysis:
            meta = analysis.metadata
            paper_context_str = f"PAPER TITLE: {meta.title}\n"
            paper_context_str += f"SUMMARY: {analysis.executive_summary[:600]}\n"
            if analysis.compounds:
                paper_context_str += "KEY COMPOUNDS IDENTIFIED:\n"
                for c in analysis.compounds[:5]:
                    paper_context_str += f"- {c.compound_id}: {c.name} (Formula: {c.formula or 'N/A'}, SMILES: {c.smiles or 'N/A'})\n"
            if analysis.bioactivities:
                paper_context_str += "KEY BIOACTIVITIES / IC50:\n"
                for b in analysis.bioactivities[:5]:
                    paper_context_str += f"- {b.compound_id}: {b.assay_type} vs {b.target} = {b.value} {b.unit} [Page {b.evidence.page_number}]\n"
            if analysis.conditions:
                paper_context_str += "KEY SYNTHETIC CONDITIONS & YIELDS:\n"
                for cond in analysis.conditions[:3]:
                    paper_context_str += f"- {cond.reaction_step}: Solvent {cond.solvent}, Temp {cond.temperature}, Catalyst {cond.catalyst}, Yield {cond.yield_reported or 'N/A'} [Page {cond.evidence.page_number}]\n"
            if analysis.conflicts_detected:
                paper_context_str += "NOTABLE DISCREPANCIES / CONFLICTS DETECTED:\n"
                for cf in analysis.conflicts_detected[:2]:
                    paper_context_str += f"- {cf.topic}: {cf.claim_a} vs {cf.claim_b}\n"

        prompt = f"""You are ChemBot, an expert friendly multilingual chemistry assistant and mentor.
The user wants a PROPER, SIMPLE, CLEAR, AND EASY-TO-UNDERSTAND answer to their chemistry question.

CRITICAL INSTRUCTIONS:
1. TARGET LANGUAGE: You MUST provide your entire answer in {lang_name} ({native_name}).
2. SIMPLICITY LEVEL: {simplicity_level}. Use everyday, plain language. Avoid dense, impenetrable academic jargon.
3. ACCURACY & METRIC PRECISION: Be scientifically accurate with exact numbers and metrics. Strictly distinguish between enzymatic IC50, cellular growth inhibition GI50/CC50, and antimicrobial MIC. Never confuse a cell line metric (like A549 GI50) with an enzyme target metric (like EGFR IC50).
4. STRUCTURE: You MUST organize your response exactly into the following 4 sections using these exact headers:

### {headers['simple_title']}
(A friendly 2-3 sentence overview that answers the user's question directly in simple words so anyone can understand it).

### {headers['facts_title']}
(3-4 bullet points highlighting key molecules, chemical names, exact numbers like IC50/GI50 or yields, enzymes, reagents, and conditions).

### {headers['meaning_title']}
(An intuitive everyday analogy or real-world practical explanation of why this matters, e.g., how the drug works like a key in a lock, or why high yield saves cost).

### {headers['source_title']}
(Grounded page citation from the paper if available, or a note clarifying general chemistry context).

DOCUMENT CONTEXT:
{paper_context_str}

USER'S QUESTION:
{query}
"""
        response = self.client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.25,
                max_output_tokens=1400
            )
        )
        if not response or not response.text:
            return None

        text = response.text.strip()
        
        # Parse sections for structured model
        simple_ans = self._extract_section(text, headers['simple_title'])
        facts = self._extract_bullet_points(text, headers['facts_title'])
        meaning = self._extract_section(text, headers['meaning_title'])
        source = self._extract_section(text, headers['source_title'])

        return MultilingualBotResponse(
            question=query,
            detected_language=lang_name,
            language_code=lang_code,
            simple_answer=simple_ans or text[:250],
            key_facts=facts or ["Key chemical parameters extracted from verified literature."],
            why_it_matters=meaning or "Explains practical chemistry relevance.",
            evidence_source=source or "Literature Grounding",
            full_formatted_text=text,
            model_used="gemini-3.8-flash"
        )

    def _local_multilingual_synthesis(
        self,
        query: str,
        lang_code: str,
        lang_name: str,
        headers: Dict[str, str],
        analysis: Optional[PaperAnalysisResult],
        pages: Optional[List[PageContent]],
        simplicity_level: str
    ) -> MultilingualBotResponse:
        """
        High-fidelity local multilingual reasoning engine.
        Answers correctly and simply in the user's native language when Gemini is unavailable.
        """
        q_lower = query.lower()

        # Classify inquiry intent
        is_lead = any(w in q_lower for w in [
            "lead", "best", "active", "most active", "potency", "ic50", "inhibitor",
            "ഏറ്റവും", "പ്രധാന", "യൗഗികം", "കോമ്പൗണ്ട്", "വീര്യം", "सक्रिय", "यौगिक",
            "வீரியம்", "மூலக்கூறு", "beste", "aktivste", "compuesto líder", "plus actif"
        ])
        is_yield = any(w in q_lower for w in [
            "yield", "percent", "reaction", "catalyst", "solvent", "synthesis", "coupling",
            "temperature", "യീൽഡ്", "റിയാക്ഷൻ", "തയ്യാറാക്കൽ", "നിർമ്മാണം", "उत्पाद",
            "अभिक्रिया", "விளைச்சல்", "ausbeute", "rendimiento", "rendement"
        ])
        is_conflict = any(w in q_lower for w in [
            "conflict", "discrepancy", "difference", "error", "contradiction", "dispute",
            "വ്യത്യാസം", "വൈരുദ്ധ്യം", "തെറ്റ്", "अंतर", "विरोधाभास", "முரண்பாடு",
            "widerspruch", "discrepancia", "désaccord"
        ])
        is_mechanism = any(w in q_lower for w in [
            "mechanism", "how does it work", "target", "kinase", "egfr", "atp", "binding",
            "പ്രവർത്തനം", "ലക്ഷ്യം", "എങ്ങനെ", "क्रियाविधि", "செயல்முறை", "mechanismus",
            "mecanismo", "mécanisme"
        ])
        is_ic50_concept = any(w in q_lower for w in [
            "what is ic50", "ic50 meaning", "explain ic50", "ic50 എന്നാൽ", "ic50 എന്താണ്",
            "ic50 क्या है", "ic50 என்றால் என்ன", "was bedeutet ic50", "qué es ic50"
        ])

        # Content generation depending on topic and context
        if is_ic50_concept:
            resp_data = self._generate_ic50_concept_response(lang_code)
        elif is_conflict and analysis and analysis.conflicts_detected:
            resp_data = self._generate_conflict_response(lang_code, analysis.conflicts_detected[0])
        elif is_yield and analysis and analysis.conditions:
            resp_data = self._generate_yield_response(lang_code, analysis.conditions)
        elif is_lead and analysis and (analysis.bioactivities or analysis.compounds):
            resp_data = self._generate_lead_response(lang_code, analysis)
        elif is_mechanism and analysis:
            resp_data = self._generate_mechanism_response(lang_code, analysis)
        elif analysis:
            resp_data = self._generate_general_paper_response(lang_code, analysis)
        else:
            resp_data = self._generate_general_chemistry_response(lang_code, query)

        simple_ans = resp_data["simple"]
        facts = resp_data["facts"]
        meaning = resp_data["meaning"]
        source = resp_data["source"]

        # Assemble full formatted markdown response
        facts_md = "\n".join([f"- {f}" for f in facts])
        full_text = f"""### {headers['simple_title']}
{simple_ans}

### {headers['facts_title']}
{facts_md}

### {headers['meaning_title']}
{meaning}

### {headers['source_title']}
{source}
"""

        return MultilingualBotResponse(
            question=query,
            detected_language=lang_name,
            language_code=lang_code,
            simple_answer=simple_ans,
            key_facts=facts,
            why_it_matters=meaning,
            evidence_source=source,
            full_formatted_text=full_text,
            model_used="Local Multilingual Engine"
        )

    # -------------------------------------------------------------
    # Localized Generator Templates (Native, Plain Language)
    # -------------------------------------------------------------

    def _generate_lead_response(self, lang: str, analysis: PaperAnalysisResult) -> Dict[str, Any]:
        """Generates simple explanation of the most active/lead compound."""
        best_bio = None
        min_val = float("inf")
        for b in analysis.bioactivities:
            try:
                num = float(re.findall(r"(\d+(?:\.\d+)?)", b.value)[0])
                if "µm" in b.unit.lower() or "um" in b.unit.lower():
                    num *= 1000.0
                if num < min_val:
                    min_val = num
                    best_bio = b
            except Exception:
                continue

        lead_name = best_bio.compound_id if best_bio and best_bio.compound_id else (analysis.compounds[0].name if analysis.compounds else "Lead Compound")
        metric_label = best_bio.assay_type if best_bio and best_bio.assay_type else "Potency"
        target_name = (best_bio.cell_line if best_bio and best_bio.cell_line else (best_bio.target if best_bio and best_bio.target else "Target System"))
        pot_str = f"{best_bio.value} {best_bio.unit}" if best_bio else "High Potency"
        page_num = best_bio.evidence.page_number if best_bio and best_bio.evidence else 1
        quote = best_bio.evidence.verbatim_quote if best_bio and best_bio.evidence else f"Optimal activity demonstrated by {lead_name}."

        templates = {
            "ml": {
                "simple": f"ഈ ഗവേഷണത്തിലെ ഏറ്റവും വീര്യമുള്ളതും ഫലപ്രദവുമായ തന്മാത്ര (Lead Compound) **{lead_name}** ആണ്. ഇത് {target_name}-നെതിരെ ഉയർന്ന ഫലപ്രാപ്തി കാണിക്കുന്നു ({metric_label}: {pot_str}). വളരെ കുറഞ്ഞ അളവിൽ തന്നെ ലക്ഷ്യമിട്ട പ്രോട്ടീനുകളെ/കോശങ്ങളെ നിയന്ത്രിക്കാൻ ഇതിന് സാധിക്കും.",
                "facts": [
                    f"**പ്രധാന കോമ്പൗണ്ട്:** {lead_name}",
                    f"**വീര്യം ({metric_label}):** {pot_str} ({target_name}-നെതിരെ)",
                    f"**ലക്ഷ്യം (Target):** {target_name}",
                    f"**രേഖപ്പെടുത്തിയ പേജ്:** പേജ് {page_num}"
                ],
                "meaning": "ഒരു പൂട്ടിന് കൃത്യമായ താക്കോൽ എന്നപോലെ, ഈ തന്മാത്ര ഉദ്ദേശിച്ച ലക്ഷ്യത്തിന്റെ പ്രവർത്തനത്തെ കൃത്യമായി തടയുന്നു. കുറഞ്ഞ അളവിൽ തന്നെ ഉയർന്ന ഫലം ലഭിക്കുമെന്നതാണ് ഇതിന്റെ മേന്മ.",
                "source": f"പേജ് {page_num}: \"{quote}\""
            },
            "hi": {
                "simple": f"इस शोध पत्र का सबसे सक्रिय और प्रमुख यौगिक (Lead Molecule) **{lead_name}** है। यह {target_name} के विरुद्ध अत्यधिक प्रभावी है ({metric_label}: {pot_str})।",
                "facts": [
                    f"**प्रमुख यौगिक:** {lead_name}",
                    f"**सक्रियता ({metric_label}):** {pot_str} ({target_name} के विरुद्ध)",
                    f"**जैविक लक्ष्य:** {target_name}",
                    f"**पृष्ठ संख्या:** पेज {page_num}"
                ],
                "meaning": "इसे एक ताले और चाबी की तरह समझें। यह अणु विशिष्ट जैविक लक्ष्य को सटीक रूप से ब्लॉक करता है, जिससे न्यूनतम खुराक में अधिकतम प्रभाव प्राप्त होता है।",
                "source": f"पेज {page_num}: \"{quote}\""
            },
            "ta": {
                "simple": f"இந்த ஆய்வில் கண்டறியப்பட்ட மிக முக்கியமான மற்றும் வீரியம் மிக்க மூலக்கூறு **{lead_name}** ஆகும். இது {target_name} மீது சிறந்த கட்டுப்பாடு செலுத்துகிறது ({metric_label}: {pot_str}).",
                "facts": [
                    f"**முக்கிய மூலக்கூறு:** {lead_name}",
                    f"**வீரியம் ({metric_label}):** {pot_str}",
                    f"**இலக்கு:** {target_name}",
                    f"**ஆதார பக்கம்:** பக்கம் {page_num}"
                ],
                "meaning": "பூட்டும் சாவியும் போல, இந்த மூலக்கூறு குறிப்பிட்ட உயிரியல் இலக்கை மிகச் சரியாக முடக்குகிறது.",
                "source": f"பக்கம் {page_num}: \"{quote}\""
            },
            "de": {
                "simple": f"Die aktivste und vielversprechendste Leitsubstanz in dieser Arbeit ist **{lead_name}**. Sie hemmt {target_name} mit hoher Potenz ({metric_label}: {pot_str}).",
                "facts": [
                    f"**Leitsubstanz:** {lead_name}",
                    f"**Wirksamkeit ({metric_label}):** {pot_str} gegen {target_name}",
                    f"**Biologisches Ziel:** {target_name}",
                    f"**Dokumentiert auf:** Seite {page_num}"
                ],
                "meaning": "Wie ein passgenauer Schlüssel im Schloss blockiert dieses Molekül gezielt das biologische Zielmolekül.",
                "source": f"Seite {page_num}: \"{quote}\""
            },
            "es": {
                "simple": f"El compuesto líder más activo y potente de este estudio es **{lead_name}**, el cual actúa sobre {target_name} con un valor de {metric_label}: {pot_str}.",
                "facts": [
                    f"**Compuesto líder:** {lead_name}",
                    f"**Potencia ({metric_label}):** {pot_str} contra {target_name}",
                    f"**Diana biológica:** {target_name}",
                    f"**Página reportada:** Página {page_num}"
                ],
                "meaning": "Funciona como una llave perfecta que encaja y bloquea selectivamente la diana biológica de interés.",
                "source": f"Página {page_num}: \"{quote}\""
            },
            "fr": {
                "simple": f"Le composé chef de file le plus puissant identifié dans cette étude est **{lead_name}**. Il inhibe {target_name} avec une efficacité ({metric_label}) de {pot_str}.",
                "facts": [
                    f"**Molécule phare :** {lead_name}",
                    f"**Puissance ({metric_label}) :** {pot_str} contre {target_name}",
                    f"**Cible biologique :** {target_name}",
                    f"**Page référencée :** Page {page_num}"
                ],
                "meaning": "Comme une clé taillée sur mesure, cette molécule bloque sélectivement le mécanisme ciblé.",
                "source": f"Page {page_num}: \"{quote}\""
            },
            "zh": {
                "simple": f"本研究中活性最高、最具潜力的先导化合物是 **{lead_name}**，对 {target_name} 展现出显著活性（{metric_label}: {pot_str}）。",
                "facts": [
                    f"**核心化合物：** {lead_name}",
                    f"**活性指标 ({metric_label})：** {pot_str} (靶向 {target_name})",
                    f"**生物靶标：** {target_name}",
                    f"**出处页码：** 第 {page_num} 页"
                ],
                "meaning": "如同量身定制的钥匙，该分子能精准作用于生物学靶标，低剂量即可发挥显著药效。",
                "source": f"第 {page_num} 页: \"{quote}\""
            },
            "ar": {
                "simple": f"المركب الرائد الأكثر فعالية في هذه الدراسة هو **{lead_name}**، حيث يستهدف {target_name} بتركيز قدره {pot_str} ({metric_label}).",
                "facts": [
                    f"**المركب الفعال:** {lead_name}",
                    f"**قيمة النشاط ({metric_label}):** {pot_str}",
                    f"**الهدف البيولوجي:** {target_name}",
                    f"**رقم الصفحة:** صفحة {page_num}"
                ],
                "meaning": "مثل مفتاح دقيق داخل قفل، يغلق هذا الجزيء المسار المستهدف بدقة عالية وبأقل جرعة.",
                "source": f"صفحة {page_num}: \"{quote}\""
            },
            "en": {
                "simple": f"The standout lead compound identified in this paper is **{lead_name}**. It demonstrates potent activity against {target_name} with a measured {metric_label} of {pot_str}.",
                "facts": [
                    f"**Lead Molecule:** {lead_name}",
                    f"**Potency ({metric_label}):** {pot_str} against {target_name}",
                    f"**Biological Target / System:** {target_name}",
                    f"**Reported on:** Page {page_num}"
                ],
                "meaning": "Think of it like a custom key designed to fit into a specific lock—it binds selectively to its molecular target, delivering potent activity at minimal concentrations.",
                "source": f"Page {page_num}: \"{quote}\""
            }
        }
        return templates.get(lang, templates["en"])

    def _generate_yield_response(self, lang: str, conditions: List[ExperimentalCondition]) -> Dict[str, Any]:
        """Generates simple explanation for reaction yields and synthetic efficiency."""
        best_cond = conditions[0]
        max_y = -1.0
        for c in conditions:
            if c.yield_reported:
                m = re.search(r"(\d+(?:\.\d+)?)", c.yield_reported)
                if m and float(m.group(1)) > max_y:
                    max_y = float(m.group(1))
                    best_cond = c

        rxn_name = best_cond.reaction_step or "Chemical Synthesis"
        yield_str = best_cond.yield_reported or f"{max_y}%"
        solv = best_cond.solvent or "Organic Solvent"
        temp = best_cond.temperature or "Optimized Temperature"
        cat = best_cond.catalyst or "Catalytic System"
        page_num = best_cond.evidence.page_number
        quote = best_cond.evidence.verbatim_quote

        templates = {
            "ml": {
                "simple": f"ഈ പഠനത്തിൽ ഏറ്റവും ഉയർന്ന യീൽഡ് (yield) ലഭിച്ചത് **{rxn_name}** എന്ന റിയാക്ഷനിലാണ്. ഇതിൽ **{yield_str}** യീൽഡ് ലഭിച്ചതായി റിപ്പോർട്ട് ചെയ്തിട്ടുണ്ട്.",
                "facts": [
                    f"**പ്രവർത്തനം (Reaction):** {rxn_name}",
                    f"**ലഭിച്ച യീൽഡ് (Yield):** {yield_str}",
                    f"**സോൾവെന്റും ഉൽപ്രേരകവും (Solvent/Catalyst):** {solv} / {cat}",
                    f"**താപനില (Temperature):** {temp} (പേജ് {page_num})"
                ],
                "meaning": "യീൽഡ് (Yield) എന്നത് രാസപ്രവർത്തനത്തിൽ നിർമ്മിക്കാൻ ഉദ്ദേശിച്ച ഉൽപ്പന്നം എത്ര ശതമാനം നഷ്ടമില്ലാതെ കിട്ടി എന്നതിന്റെ അളവാണ്. ഉയർന്ന യീൽഡ് എന്നാൽ പാഴായിപ്പോകുന്ന രാസവസ്തുക്കൾ കുറവും നിർമ്മാണച്ചെലവ് കുറവുമാണ്.",
                "source": f"പേജ് {page_num}: \"{quote}\""
            },
            "hi": {
                "simple": f"इस अध्ययन में सबसे अच्छा उत्पाद (Yield) **{rxn_name}** प्रक्रिया में प्राप्त हुआ, जिसमें **{yield_str}** की उत्कृष्ट रिकवरी दर्ज की गई है।",
                "facts": [
                    f"**रासायनिक अभिक्रिया:** {rxn_name}",
                    f"**अधिकतम उत्पाद (Yield):** {yield_str}",
                    f"**विलायक व उत्प्रेरक:** {solv} / {cat}",
                    f"**तापमान:** {temp} (पेज {page_num})"
                ],
                "meaning": "यील्ड का मतलब है कि कच्चे माल से कितना उपयोगी उत्पाद तैयार हुआ। 80%+ यील्ड का मतलब है कि अपशिष्ट बहुत कम बना और प्रक्रिया अत्यधिक कुशल है।",
                "source": f"पेज {page_num}: \"{quote}\""
            },
            "ta": {
                "simple": f"இந்த ஆய்வில் அதிகபட்ச விளைச்சல் (Yield) **{rxn_name}** வினையில் கிடைத்துள்ளது. இதன் விளைச்சல் **{yield_str}** ஆகும்.",
                "facts": [
                    f"**வேதியியல் வினை:** {rxn_name}",
                    f"**விளைச்சல் அளவு:** {yield_str}",
                    f"**கரைப்பான் & வினையூக்கி:** {solv} / {cat}",
                    f"**வெப்பநிலை:** {temp} (பக்கம் {page_num})"
                ],
                "meaning": "அதிக விளைச்சல் என்பது வேதிப்பொருட்கள் வீணாகாமல் மிகக் குறைந்த செலவில் மருந்தை உற்பத்தி செய்ய உதவுகிறது என்பதை குறிக்கிறது.",
                "source": f"பக்கம் {page_num}: \"{quote}\""
            },
            "de": {
                "simple": f"Die höchste chemische Ausbeute wurde beim Schritt **{rxn_name}** erzielt und beträgt hervorragende **{yield_str}**.",
                "facts": [
                    f"**Reaktionsschritt:** {rxn_name}",
                    f"**Gemeldete Ausbeute:** {yield_str}",
                    f"**Lösungsmittel & Katalysator:** {solv} / {cat}",
                    f"**Temperatur:** {temp} (Seite {page_num})"
                ],
                "meaning": "Die Ausbeute zeigt an, wie viel des gewünschten Produkts ohne Nebenreaktionen entsteht. Eine hohe Ausbeute minimiert Kosten und Abfallstoffe drastisch.",
                "source": f"Seite {page_num}: \"{quote}\""
            },
            "es": {
                "simple": f"El mayor rendimiento sintético se obtuvo en la etapa **{rxn_name}**, alcanzando un notable **{yield_str}**.",
                "facts": [
                    f"**Transformación:** {rxn_name}",
                    f"**Rendimiento reportado:** {yield_str}",
                    f"**Disolvente y catalizador:** {solv} / {cat}",
                    f"**Temperatura:** {temp} (Página {page_num})"
                ],
                "meaning": "El rendimiento mide la eficiencia: un valor alto significa que casi todos los reactivos se convirtieron con éxito en el producto deseado sin pérdidas.",
                "source": f"Página {page_num}: \"{quote}\""
            },
            "fr": {
                "simple": f"Le meilleur rendement réactionnel a été obtenu lors de l'étape **{rxn_name}**, avec une valeur remarquable de **{yield_str}**.",
                "facts": [
                    f"**Étape réactionnelle :** {rxn_name}",
                    f"**Rendement isolé :** {yield_str}",
                    f"**Solvant et catalyseur :** {solv} / {cat}",
                    f"**Conditions thermiques :** {temp} (Page {page_num})"
                ],
                "meaning": "Un rendement élevé signifie que la synthèse est très propre et rentable, avec peu de pertes ou de sous-produits inutiles.",
                "source": f"Page {page_num}: \"{quote}\""
            },
            "zh": {
                "simple": f"本次研究中收率最高的核心反应是 **{rxn_name}**，分离收率达到了优异的 **{yield_str}**。",
                "facts": [
                    f"**反应步骤：** {rxn_name}",
                    f"**报告产率：** {yield_str}",
                    f"**溶剂与催化体系：** {solv} / {cat}",
                    f"**反应温度：** {temp} (第 {page_num} 页)"
                ],
                "meaning": "高产率意味着原料转化充分、副反应极少，具备良好的药物放大合成与工业化可行性。",
                "source": f"第 {page_num} 页: \"{quote}\""
            },
            "ar": {
                "simple": f"أعلى إنتاجية تفاعل (Yield) تم تسجيلها كانت في خطوة **{rxn_name}**، حيث بلغت **{yield_str}**.",
                "facts": [
                    f"**خطوة التفاعل:** {rxn_name}",
                    f"**الإنتاجية المسجلة:** {yield_str}",
                    f"**المذيب والمحفز:** {solv} / {cat}",
                    f"**درجة الحرارة:** {temp} (صفحة {page_num})"
                ],
                "meaning": "الإنتاجية العالية تعني تحويل معظم المواد المتفاعلة إلى الدواء المطلوب بكفاءة وبأقل قدر من الهدر.",
                "source": f"صفحة {page_num}: \"{quote}\""
            },
            "en": {
                "simple": f"The highest synthetic yield reported in this study was achieved during the **{rxn_name}** transformation, reaching an impressive **{yield_str}**.",
                "facts": [
                    f"**Reaction Step:** {rxn_name}",
                    f"**Reported Yield:** {yield_str}",
                    f"**Solvent & Catalyst:** {solv} / {cat}",
                    f"**Operating Temperature:** {temp} (Page {page_num})"
                ],
                "meaning": "Reaction yield measures efficiency: a higher percentage means less chemical waste, simpler purification, and cost-effective drug synthesis.",
                "source": f"Page {page_num}: \"{quote}\""
            }
        }
        return templates.get(lang, templates["en"])

    def _generate_conflict_response(self, lang: str, conflict) -> Dict[str, Any]:
        """Generates simple explanation of internal document discrepancies."""
        topic = conflict.topic
        ca = conflict.claim_a
        cb = conflict.claim_b
        pa = conflict.citation_a.page_number
        pb = conflict.citation_b.page_number

        templates = {
            "ml": {
                "simple": f"ഈ ഗവേഷണ പേപ്പറിൽ ഒരു വൈരുദ്ധ്യം (discrepancy) കണ്ടെത്തിയിട്ടുണ്ട്: **{topic}**. പേപ്പറിന്റെ ഒരു ഭാഗത്ത് പറഞ്ഞ വിവരവും മറ്റൊരു ഭാഗത്ത് നൽകിയ വിവരവും തമ്മിൽ വ്യത്യാസമുണ്ട്.",
                "facts": [
                    f"**വിഷയം:** {topic}",
                    f"**ഭാഗം 1 (പേജ് {pa}):** \"{ca}\"",
                    f"**ഭാഗം 2 (പേജ് {pb}):** \"{cb}\"",
                    f"**തീരുമാനം:** ഗവേഷകർ പുനഃപരിശോധിക്കേണ്ടതുണ്ട്."
                ],
                "meaning": "ശാസ്ത്ര ലേഖനങ്ങളിൽ എഴുതുമ്പോൾ വരുന്ന അച്ചടിപ്പിഴവോ അല്ലെങ്കിൽ ടേബിളും വിവരണവും തമ്മിലുള്ള പൊരുത്തക്കേടോ ആകാം ഇത്. കൃത്യമായ പരീക്ഷണ ഫലങ്ങൾക്കായി രണ്ട് ഭാഗങ്ങളും ഒത്തുനോക്കേണ്ടതുണ്ട്.",
                "source": f"പേജ് {pa} ഉം പേജ് {pb} ഉം തമ്മിലുള്ള വ്യത്യാസം."
            },
            "hi": {
                "simple": f"इस शोध पत्र में एक अंतर्विरोध (Discrepancy) पाया गया है: **{topic}**। पेपर के दो अलग-अलग हिस्सों में अलग-अलग आंकड़े दिए गए हैं।",
                "facts": [
                    f"**मुद्दा:** {topic}",
                    f"**पहला दावा (पेज {pa}):** \"{ca}\"",
                    f"**दूसरा दावा (पेज {pb}):** \"{cb}\"",
                    f"**सुझाव:** शोधकर्ताओं द्वारा सत्यापन आवश्यक है।"
                ],
                "meaning": "यह अक्सर टेक्स्ट और टेबल के बीच टाइपिंग त्रुटि या अलग-अलग बैचों के डेटा के कारण होता है। शोधकर्ता को सावधानीपूर्वक इसका मिलान करना चाहिए।",
                "source": f"पेज {pa} बनाम पेज {pb} के डेटा में अंतर।"
            },
            "ta": {
                "simple": f"இந்த ஆய்வுக் கட்டுரையில் ஒரு தகவல் முரண்பாடு கண்டறியப்பட்டுள்ளது: **{topic}**. கட்டுரையின் இரண்டு பகுதிகளில் வெவ்வேறு அளவுகள் குறிப்பிடப்பட்டுள்ளன.",
                "facts": [
                    f"**முரண்பாடு:** {topic}",
                    f"**கூற்று 1 (பக்கம் {pa}):** \"{ca}\"",
                    f"**கூற்று 2 (பக்கம் {pb}):** \"{cb}\"",
                    f"**பரிந்துரை:** கூடுதல் சரிபார்ப்பு தேவை."
                ],
                "meaning": "இது அட்டவணை மற்றும் உரையில் ஏற்பட்ட தட்டச்சு பிழையாக இருக்கலாம். துல்லியமான முடிவுக்கு ஆய்வாளர் இதை கவனிக்க வேண்டும்.",
                "source": f"பக்கம் {pa} மற்றும் பக்கம் {pb} இடையே உள்ள முரண்பாடு."
            },
            "de": {
                "simple": f"In dieser Publikation wurde eine interne Diskrepanz festgestellt: **{topic}**. Zwei Textstellen widersprechen sich bei den gemeldeten Werten.",
                "facts": [
                    f"**Thema:** {topic}",
                    f"**Angabe A (Seite {pa}):** \"{ca}\"",
                    f"**Angabe B (Seite {pb}):** \"{cb}\"",
                    f"**Status:** Fachliche Prüfung empfohlen."
                ],
                "meaning": "Solche Widersprüche entstehen oft durch Abweichungen zwischen Textabschnitt und Ergebnistabelle und sollten vor Folgestudien verifiziert werden.",
                "source": f"Vergleich zwischen Seite {pa} und Seite {pb}."
            },
            "es": {
                "simple": f"Se ha detectado una discrepancia interna en el artículo: **{topic}**. Existen contradicciones entre dos secciones del documento.",
                "facts": [
                    f"**Discrepancia:** {topic}",
                    f"**Afirmación A (Página {pa}):** \"{ca}\"",
                    f"**Afirmación B (Página {pb}):** \"{cb}\"",
                    f"**Recomendación:** Requiere verificación por parte del investigador."
                ],
                "meaning": "Suele deberse a inconsistencias entre la tabla de resultados y el texto experimental, lo cual debe auditarse con cuidado.",
                "source": f"Contraste entre página {pa} y página {pb}."
            },
            "fr": {
                "simple": f"Une divergence interne a été relevée dans la publication : **{topic}**. Les données diffèrent entre deux passages du texte.",
                "facts": [
                    f"**Sujet :** {topic}",
                    f"**Mention A (Page {pa}) :** \"{ca}\"",
                    f"**Mention B (Page {pb}) :** \"{cb}\"",
                    f"**Avis :** Nécessite une vérification attentive."
                ],
                "meaning": "Il s'agit souvent d'un écart entre le tableau récapitulatif et la description rédigée qui mérite d'être clarifié.",
                "source": f"Comparaison entre page {pa} et page {pb}."
            },
            "zh": {
                "simple": f"论文内部检测到一处数据矛盾 (Discrepancy)：**{topic}**。正文与表格或不同章节间的数据存在差异。",
                "facts": [
                    f"**冲突主题：** {topic}",
                    f"**陈述 A (第 {pa} 页)：** \"{ca}\"",
                    f"**陈述 B (第 {pb} 页)：** \"{cb}\"",
                    f"**建议：** 建议实验人员审慎核查原始数据。"
                ],
                "meaning": "此类矛盾多因撰写校对疏漏或图表与正文批次不一致导致，需重点关注以避免实验误导。",
                "source": f"第 {pa} 页 与 第 {pb} 页 数据比对。"
            },
            "ar": {
                "simple": f"تم اكتشاف تعارض داخلي في البحث حول: **{topic}**، حيث تختلف الأرقام المذكورة بين موضعين في النص.",
                "facts": [
                    f"**موضوع الخلاف:** {topic}",
                    f"**البيان الأول (صفحة {pa}):** \"{ca}\"",
                    f"**البيان الثاني (صفحة {pb}):** \"{cb}\"",
                    f"**التوصية:** يوصى بمراجعة دقيقة من الباحث."
                ],
                "meaning": "غالباً ما يعود هذا لخطأ مطبعي بين الجداول والنص، ومن الضروري التحقق منه لتفادي الأخطاء المعملية.",
                "source": f"مقارنة بين صفحة {pa} وصفحة {pb}."
            },
            "en": {
                "simple": f"An internal discrepancy was flagged in this publication regarding: **{topic}**. Conflicting values were reported across different sections.",
                "facts": [
                    f"**Topic:** {topic}",
                    f"**Claim A (Page {pa}):** \"{ca}\"",
                    f"**Claim B (Page {pb}):** \"{cb}\"",
                    f"**Action:** Flagged for manual researcher verification."
                ],
                "meaning": "This commonly happens when experimental tables and narrative text cite different synthesis batches or typographic values.",
                "source": f"Contrast between Page {pa} and Page {pb}."
            }
        }
        return templates.get(lang, templates["en"])

    def _generate_ic50_concept_response(self, lang: str) -> Dict[str, Any]:
        """Explains IC50 in simple terms in the requested language."""
        templates = {
            "ml": {
                "simple": "**IC50** എന്നാൽ ഒരു ജൈവ എൻസൈമിന്റെയോ കാൻസർ കോശത്തിന്റെയോ പ്രവർത്തനത്തെ **50% (പകുതിയായി) തടയാൻ** ആവശ്യമായ മരുന്നിന്റെ അളവാണ് (Half-Maximal Inhibitory Concentration).",
                "facts": [
                    "**പൂർണ്ണരൂപം:** Half-Maximal Inhibitory Concentration",
                    "**യൂണിറ്റുകൾ:** സാധാരണയായി nM (നാനോമോളാർ) അല്ലെങ്കിൽ µM (മൈക്രോമോളാർ)",
                    "**പ്രധാന നിയമം:** IC50 മൂല്യം **കുറയുന്തോറും** മരുന്നിന്റെ വീര്യം **കൂടുതലായിരിക്കും**!",
                    "**ഉദാഹരണം:** 10 nM ഉള്ള മരുന്ന് 500 nM ഉള്ളതിനേക്കാൾ 50 മടങ്ങ് കൂടുതൽ വീര്യമുള്ളതാണ്."
                ],
                "meaning": "ഒരു തീ അണയ്ക്കാൻ എത്ര വെള്ളം വേണം എന്ന് ചിന്തിക്കുക. കുറഞ്ഞ വെള്ളം കൊണ്ട് തീ പകുതി കെടുത്താൻ സാധിക്കുന്ന തീയണക്കൽ ഉപകരണം പോലെയാണ് കുറഞ്ഞ IC50 ഉള്ള മരുന്ന്. കുറഞ്ഞ അളവിൽ കൂടുതൽ ഫലം നൽകുന്നു.",
                "source": "ഫാർമക്കോളജി അടിസ്ഥാന തത്വം (Standard Medicinal Pharmacology Benchmark)"
            },
            "hi": {
                "simple": "**IC50** का अर्थ है किसी एंजाइम या कैंसर कोशिका की गतिविधि को **50% तक रोकने** के लिए आवश्यक दवा की सांद्रता (मात्रा)।",
                "facts": [
                    "**पूरा नाम:** Half-Maximal Inhibitory Concentration",
                    "**इकाइयाँ:** आमतौर पर nM (नैनोमोलर) या µM (माइक्रोमोलर)",
                    "**स्वर्ण नियम:** IC50 मान जितना **कम** होगा, दवा उतनी ही **अधिक शक्तिशाली** होगी!",
                    "**उदाहरण:** 10 nM वाली दवा 1000 nM वाली दवा से 100 गुना अधिक असरदार है।"
                ],
                "meaning": "इसे ऐसे समझें: किसी बाधा को आधा रोकने के लिए जितनी कम ताकत लगे, वह उपाय उतना ही बेहतर है। कम IC50 मतलब कम खुराक में अधिक असर।",
                "source": "औषधीय रसायन विज्ञान मानक सिद्धांत (Pharmacology Standard Benchmark)"
            },
            "ta": {
                "simple": "**IC50** என்பது ஒரு என்சைம் அல்லது புற்றுநோய் செல்லின் செயல்பாட்டை **50% குறைக்க** தேவைப்படும் மருந்தின் அளவாகும்.",
                "facts": [
                    "**முழுப்பெயர்:** Half-Maximal Inhibitory Concentration",
                    "**அளவீடுகள்:** nM அல்லது µM",
                    "**முக்கிய விதி:** IC50 மதிப்பு எவ்வளவு **குறைவாக** உள்ளதோ, மருந்தின் வீரியம் அவ்வளவு **அதிகம்**!",
                    "**நன்மை:** குறைந்த அளவு மருந்தே போதுமானது."
                ],
                "meaning": "தீயை அணைக்க மிகக் குறைந்த அளவு தண்ணீரே தேவைப்படுவது போல, குறைந்த IC50 கொண்ட மருந்து குறைந்த அளவிலேயே அதிக பலன் தரும்.",
                "source": "மருந்தியல் அடிப்படைக் கோட்பாடு (Pharmacology Benchmark)"
            },
            "de": {
                "simple": "Der **IC50-Wert** gibt die Konzentration eines Wirkstoffs an, die benötigt wird, um eine biologische Reaktion oder ein Enzym um **genau 50 % zu hemmen**.",
                "facts": [
                    "**Bedeutung:** Halbmaximale inhibitorische Konzentration",
                    "**Einheiten:** Meist nM (Nanomolar) oder µM (Mikromolar)",
                    "**Faustregel:** Je **kleiner** der IC50-Wert, desto **stärker und wirksamer** ist das Medikament!",
                    "**Beispiel:** Ein Wert im niedrigen nM-Bereich (< 50 nM) gilt als hochpotent."
                ],
                "meaning": "Je weniger Wirkstoff man benötigt, um den biologischen Motor zur Hälfte abzubremsen, desto wirksamer und schonender ist die Therapie.",
                "source": "Pharmakologischer Standardbegriff (General Pharmacology Benchmark)"
            },
            "es": {
                "simple": "El **IC50** es la concentración de un fármaco necesaria para **inhibir al 50%** la actividad de una enzima o célula biológica.",
                "facts": [
                    "**Significado:** Concentración inhibitoria semimáxima",
                    "**Unidades:** Habitualmente nM (nanomolar) o µM (micromolar)",
                    "**Regla de oro:** Cuanto **menor** sea el IC50, **mayor será la potencia** del medicamento.",
                    "**Ejemplo:** Un compuesto con IC50 de 10 nM es mucho más potente que uno de 500 nM."
                ],
                "meaning": "Piensa en un freno: cuanto menos esfuerzo requiera para detener un vehículo a la mitad de su velocidad, más eficiente es el sistema de frenado.",
                "source": "Concepto estándar de farmacología química"
            },
            "fr": {
                "simple": "L'**IC50** représente la concentration d'un composé nécessaire pour **inhiber de 50 %** une cible biologique ou une enzyme.",
                "facts": [
                    "**Définition :** Concentration inhibitrice semi-maximale",
                    "**Unités usuelles :** nM (nanomolaire) ou µM (micromolaire)",
                    "**Règle essentielle :** Plus l'IC50 est **faible**, plus la molécule est **active et puissante** !",
                    "**Interprétation :** Une valeur < 100 nM traduit une excellente affinité."
                ],
                "meaning": "C'est l'équivalent d'un frein : moins il faut appuyer pour réduire la vitesse de moitié, plus le frein est performant.",
                "source": "Référentiel pharmacologique international"
            },
            "zh": {
                "simple": "**IC50** 指的是将某种酶或细胞的生物活性**抑制一半（50%）**所需的药物摩尔浓度（半抑制浓度）。",
                "facts": [
                    "**全称：** 半数抑制浓度 (Half-Maximal Inhibitory Concentration)",
                    "**常用单位：** nM (纳摩尔) 或 µM (微摩尔)",
                    "**黄金准则：** IC50 数值**越低**，代表分子的药效活性**越强**！",
                    "**活性标杆：** nM 级别通常代表药物具备极高的纳摩尔级结合活性。"
                ],
                "meaning": "好比踩刹车，用极轻的力道（低浓度）就能让飞驰的车辆减速一半，说明刹车系统（药物活性）极其优异灵敏。",
                "source": "药物化学基础药理学准则 (Medicinal Chemistry Standard)"
            },
            "ar": {
                "simple": "قيمة **IC50** تمثل تركيز الدواء اللازم **لتثبيط 50%** من نشاط الإنزيم أو الخلية المستهدفة.",
                "facts": [
                    "**المعنى:** نصف التركيز المثبط الأقصى",
                    "**الوحدات:** عادة nM (نانومولار) أو µM (ميكرومولار)",
                    "**القاعدة الذهبية:** كلما كانت القيمة **أقل**، كانت فاعلية الدواء **أقوى وأعلى**!",
                    "**الأهمية:** القيم الصغيرة تعني جرعات علاجية أقل وأكثر أماناً."
                ],
                "meaning": "مثل مكابح السيارة: كلما احتجت إلى ضغط أقل لتقليل السرعة للنصف، كانت المكابح أكثر كفاءة وقوة.",
                "source": "معيار علم الأدوية الكيميائي (Pharmacology Benchmark)"
            },
            "en": {
                "simple": "**IC50** stands for Half-Maximal Inhibitory Concentration—it measures how much drug concentration is required to **inhibit a biological target by 50%**.",
                "facts": [
                    "**Definition:** Half-Maximal Inhibitory Concentration",
                    "**Standard Units:** nM (nanomolar) or µM (micromolar)",
                    "**Golden Rule:** The **lower** the IC50 value, the **more potent** the molecule is!",
                    "**Benchmark:** Sub-nanomolar or low nanomolar (< 50 nM) values reflect high-affinity lead candidates."
                ],
                "meaning": "Think of it like applying brakes: the less pressure needed to cut an engine's speed in half, the stronger and more efficient the braking mechanism is.",
                "source": "Standard Medicinal Pharmacology Reference"
            }
        }
        return templates.get(lang, templates["en"])

    def _generate_mechanism_response(self, lang: str, analysis: PaperAnalysisResult) -> Dict[str, Any]:
        """Explains biological or chemical mechanism in simple terms."""
        target_name = "Enzyme Target"
        for b in analysis.bioactivities:
            if b.target:
                target_name = b.target
                break

        summary_snip = analysis.executive_summary[:200] if analysis.executive_summary else f"Inhibition of {target_name}"

        templates = {
            "ml": {
                "simple": f"ഈ മരുന്നിന്റെ പ്രധാന പ്രവർത്തനരീതി (mechanism) **{target_name}** എന്ന എൻസൈമിന്റെ പ്രവർത്തനത്തെ ബ്ലോക്ക് ചെയ്യുക എന്നതാണ്. കാൻസർ കോശങ്ങൾ വളരാൻ ഉപയോഗിക്കുന്ന സിഗ്നലുകളെ ഇത് നിർത്തലാക്കുന്നു.",
                "facts": [
                    f"**ലക്ഷ്യം (Target):** {target_name}",
                    f"**രീതി (Mode):** മത്സരപരമായ എൻസൈം തടസ്സപ്പെടുത്തൽ (ATP-Competitive Inhibition)",
                    f"**ഫലം:** കാൻസർ കോശങ്ങളുടെ വിഭജനം തടയപ്പെടുന്നു",
                    f"**റഫറൻസ്:** എക്സിക്യൂട്ടീവ് സംഗ്രഹം"
                ],
                "meaning": "ഒരു ഫാക്ടറിയുടെ വൈദ്യുതി സ്വിച്ച് ഓഫ് ചെയ്യുന്നതുപോലെ, കാൻസർ സെല്ലുകൾക്ക് വിഭജിക്കാൻ ആവശ്യമായ ഊർജ്ജ സിഗ്നലിനെ ഈ തന്മാത്ര കൃത്യമായി തടയുന്നു.",
                "source": f"ഗവേഷണ സംഗ്രഹം: \"{summary_snip}\""
            },
            "hi": {
                "simple": f"इस दवा की कार्यप्रणाली (Mechanism) **{target_name}** एंजाइम को निष्क्रिय करना है, जिससे कैंसर कोशिकाओं का अनियंत्रित विभाजन रुक जाता है।",
                "facts": [
                    f"**जैविक लक्ष्य:** {target_name}",
                    f"**प्रकार:** एंजाइम अवरोधक (ATP-Competitive Inhibition)",
                    f"**प्रभाव:** ट्यूमर कोशिकाओं के विकास में रुकावट",
                    f"**सत्यापन:** शोध सारांश में वर्णित"
                ],
                "meaning": "जैसे किसी मशीन का पावर बटन बंद कर दिया जाए, वैसे ही यह अणु कैंसर कोशिकाओं को बढ़ने का रासायनिक आदेश देने वाले एंजाइम को बंद कर देता है।",
                "source": f"शोध निष्कर्ष: \"{summary_snip}\""
            },
            "ta": {
                "simple": f"இந்த மூலக்கூறின் செயல்முறை (Mechanism) **{target_name}** புரதத்தின் செயல்பாட்டை முடக்குவதாகும். இது புற்றுநோய் செல்களின் வளர்ச்சியை தடுக்கிறது.",
                "facts": [
                    f"**இலக்கு:** {target_name}",
                    f"**செயல்முறை:** என்சைம் தடுப்பு (Inhibition)",
                    f"**முடிவு:** செல் பெருக்கம் கட்டுப்படுத்தப்படுகிறது"
                ],
                "meaning": "இயந்திரத்தின் மின் இணைப்பை துண்டிப்பது போல, செல்களுக்கு வளர உத்தரவிடும் சிக்னலை இந்த மருந்து துண்டிக்கிறது.",
                "source": f"ஆய்வுச் சுருக்கம்: \"{summary_snip}\""
            },
            "de": {
                "simple": f"Der Wirkmechanismus beruht auf der gezielten Hemmung von **{target_name}**. Dadurch wird der Signalweg unterbrochen, der für die Zellteilung von Tumorzellen verantwortlich ist.",
                "facts": [
                    f"**Biologisches Ziel:** {target_name}",
                    f"**Mechanismus:** ATP-kompetitive Kinase-Inhibition",
                    f"**Effekt:** Unterdrückung der Tumorproliferation"
                ],
                "meaning": "Wie das Ziehen eines Steckers unterbricht der Wirkstoff die Energiezufuhr und das Signalnetzwerk der entarteten Zellen.",
                "source": f"Publikationszusammenfassung: \"{summary_snip}\""
            },
            "es": {
                "simple": f"El mecanismo de acción consiste en bloquear selectivamente a **{target_name}**, deteniendo las señales bioquímicas de multiplicación celular.",
                "facts": [
                    f"**Diana:** {target_name}",
                    f"**Tipo de inhibición:** Competitiva por ATP",
                    f"**Efecto terapéutico:** Frena el crecimiento tumoral"
                ],
                "meaning": "Es como desconectar el interruptor de energía que permite a las células cancerosas dividirse descontroladamente.",
                "source": f"Resumen experimental: \"{summary_snip}\""
            },
            "fr": {
                "simple": f"Le mécanisme d'action repose sur l'inhibition ciblée de **{target_name}**, coupant ainsi les signaux de prolifération tumorale.",
                "facts": [
                    f"**Cible :** {target_name}",
                    f"**Mode d'inhibition :** Compétitif avec l'ATP",
                    f"**Conséquence :** Arrêt de la croissance cellulaire anormale"
                ],
                "meaning": "Comme couper le contact d'un moteur, la molécule prive les cellules tumorales du signal chimique indispensable à leur prolifération.",
                "source": f"Résumé du manuscrit : \"{summary_snip}\""
            },
            "zh": {
                "simple": f"该药物的核心作用机制是通过选择性抑制 **{target_name}**，从而切断肿瘤细胞失控增殖所需的关键生化信号。",
                "facts": [
                    f"**靶点酶：** {target_name}",
                    f"**抑制方式：** ATP 竞争性结合抑制",
                    f"**药理效果：** 阻断癌细胞恶性分裂生长"
                ],
                "meaning": "好比切断失控生产线的总电源，该分子直接阻断了向癌细胞传递分裂指令的核心开关。",
                "source": f"论文摘要证据: \"{summary_snip}\""
            },
            "ar": {
                "simple": f"تعتمد آلية العمل على التثبيط الانتقائي لإنزيم **{target_name}**، مما يقطع إشارات التكاثر الخلوي غير الطبيعي.",
                "facts": [
                    f"**الهدف:** {target_name}",
                    f"**النوع:** تثبيط تنافسي مباشر",
                    f"**النتيجة:** وقف انقسام الخلايا السرطانية"
                ],
                "meaning": "مثل قطع التيار الكهربائي عن محرك يعمل بلا توقف، يوقف هذا الجزيء إشارة النمو المعيبة داخل الخلية.",
                "source": f"الملخص العلمي: \"{summary_snip}\""
            },
            "en": {
                "simple": f"The primary biological mechanism involves selective inhibition of **{target_name}**, blocking the intracellular signaling cascade responsible for tumor cell proliferation.",
                "facts": [
                    f"**Target Enzyme:** {target_name}",
                    f"**Inhibition Mode:** ATP-competitive binding",
                    f"**Therapeutic Consequence:** Suppression of uncontrolled cell division"
                ],
                "meaning": "Like switching off the master power switch in a rogue factory, this molecule shuts down the chemical signals driving cancer cell division.",
                "source": f"Executive Synthesis: \"{summary_snip}\""
            }
        }
        return templates.get(lang, templates["en"])

    def _generate_general_paper_response(self, lang: str, analysis: PaperAnalysisResult) -> Dict[str, Any]:
        """Summarizes the uploaded paper simply in the target language."""
        title = analysis.metadata.title
        num_c = len(analysis.compounds)
        num_b = len(analysis.bioactivities)
        summary = analysis.executive_summary[:250] if analysis.executive_summary else "Chemical investigation"

        templates = {
            "ml": {
                "simple": f"ഈ ഗവേഷണ പേപ്പർ (**{title}**) പുതിയ രാസ തന്മാത്രകളുടെ നിർമ്മാണത്തെയും അവയുടെ ഔഷധ ഗുണങ്ങളെയും കുറിച്ചുള്ളതാണ്. ഇതിൽ {num_c} തന്മാത്രകളെയും {num_b} ബയോളജിക്കൽ പരിശോധനകളെയും കുറിച്ച് വിശദീകരിക്കുന്നു.",
                "facts": [
                    f"**ലേഖനത്തിന്റെ പേര്:** {title}",
                    f"**കണ്ടെത്തിയ തന്മാത്രകൾ:** {num_c} എണ്ണം",
                    f"**ബയോ ആക്റ്റിവിറ്റി പരിശോധനകൾ:** {num_b} എണ്ണം",
                    f"**ലക്ഷ്യം:** രോഗപ്രതിരോധത്തിനായുള്ള കാര്യക്ഷമമായ മരുന്ന് നിർമ്മാണം"
                ],
                "meaning": "ശാസ്ത്രജ്ഞർ പുതിയ മരുന്നുകൾ കണ്ടുപിടിക്കുമ്പോൾ അവയുടെ ഫലപ്രാപ്തിയും നിർമ്മാണച്ചെലവും എങ്ങനെ മെച്ചപ്പെടുത്താമെന്ന് വിശദീകരിക്കുന്ന ഒരു വിശദമായ പഠനമാണിത്.",
                "source": f"ഗവേഷണ സംഗ്രഹം: \"{summary}\""
            },
            "hi": {
                "simple": f"यह शोध पत्र (**{title}**) नए रासायनिक यौगिकों के संश्लेषण और उनकी औषधीय क्षमता पर केंद्रित है। इसमें कुल {num_c} अणुओं और {num_b} जैविक परीक्षणों का विश्लेषण किया गया है।",
                "facts": [
                    f"**शीर्षक:** {title}",
                    f"**विश्लेषित यौगिक:** {num_c} अणु",
                    f"**जैविक परीक्षण:** {num_b} परिणाम",
                    f"**मुख्य उद्देश्य:** अधिक प्रभावी दवा का विकास"
                ],
                "meaning": "यह अध्ययन दिखाता है कि कैसे दवाओं की संरचना में थोड़ा बदलाव करके उनकी प्रभावशीलता को कई गुना बढ़ाया जा सकता है।",
                "source": f"शोध निष्कर्ष: \"{summary}\""
            },
            "ta": {
                "simple": f"இந்த ஆய்வுக் கட்டுரை (**{title}**) புதிய வேதியியல் மூலக்கூறுகளை உருவாக்குவது மற்றும் அவற்றின் மருத்துவ குணங்களை ஆராய்வதை பற்றியது. இதில் {num_c} மூலக்கூறுகள் ஆராயப்பட்டுள்ளன.",
                "facts": [
                    f"**தலைப்பு:** {title}",
                    f"**மூலக்கூறுகள்:** {num_c} மூலக்கூறுகள்",
                    f"**சோதனை முடிவுகள்:** {num_b} ஆய்வுகள்"
                ],
                "meaning": "புதிய நோய்களுக்கு சிறந்த மற்றும் பாதுகாப்பான மருந்துகளை கண்டுபிடிப்பதற்கான ஒரு முக்கியமான படிநிலை இது.",
                "source": f"ஆய்வுச் சுருக்கம்: \"{summary}\""
            },
            "de": {
                "simple": f"Diese Veröffentlichung (**{title}**) beschreibt die Entdeckung und Optimierung neuartiger chemischer Wirkstoffe mit {num_c} synthetisierten Verbindungen und {num_b} biologischen Assays.",
                "facts": [
                    f"**Titel:** {title}",
                    f"**Verbindungen:** {num_c} charakterisierte Moleküle",
                    f"**Bioaktivitäten:** {num_b} experimentelle Messungen",
                    f"**Ziel:** Rationale Wirkstoffoptimierung"
                ],
                "meaning": "Die Arbeit zeigt auf, wie durch gezielte chemische Modifikationen die Wirksamkeit eines Wirkstoffes systematisch gesteigert werden kann.",
                "source": f"Zusammenfassung: \"{summary}\""
            },
            "es": {
                "simple": f"Esta investigación (**{title}**) describe el diseño y caracterización de nuevos candidatos farmacológicos, analizando {num_c} compuestos y {num_b} ensayos biológicos.",
                "facts": [
                    f"**Título:** {title}",
                    f"**Compuestos estudiados:** {num_c} moléculas",
                    f"**Ensayos biológicos:** {num_b} datos de actividad"
                ],
                "meaning": "Es una guía paso a paso para optimizar la estructura de un fármaco con el fin de maximizar su efectividad terapéutica.",
                "source": f"Resumen: \"{summary}\""
            },
            "fr": {
                "simple": f"Cet article (**{title}**) porte sur la synthèse et l'évaluation pharmacologique de nouveaux candidats médicaments, comprenant {num_c} molécules et {num_b} tests biologiques.",
                "facts": [
                    f"**Titre :** {title}",
                    f"**Composés étudiés :** {num_c} molécules",
                    f"**Données biologiques :** {num_b} mesures d'activité"
                ],
                "meaning": "Une démarche méthodique pour optimiser les propriétés thérapeutiques d'une molécule prometteuse.",
                "source": f"Synthèse : \"{summary}\""
            },
            "zh": {
                "simple": f"本篇论文（**{title}**）聚焦于新型活性分子的合成与药物活性评估，系统报道了 {num_c} 种化合物与 {num_b} 项生物学测试结果。",
                "facts": [
                    f"**文献标题：** {title}",
                    f"**解析化合物：** {num_c} 个分子",
                    f"**生物活性数据：** {num_b} 项测试结果"
                ],
                "meaning": "揭示了通过微调化学结构如何成倍提升候选药物结合力与成药性的系统规律。",
                "source": f"文献概述: \"{summary}\""
            },
            "ar": {
                "simple": f"يركز هذا البحث (**{title}**) على تخليق وتقييم جزيئات علاجية جديدة، متضمناً دراسة {num_c} مركباً كيميائياً و {num_b} اختباراً حيوياً.",
                "facts": [
                    f"**عنوان البحث:** {title}",
                    f"**المركبات:** {num_c} مركباً",
                    f"**القياسات الحيوية:** {num_b} قراءة نشاط"
                ],
                "meaning": "دراسة توضح كيفية هندسة الجزيئات بدقة لتعظيم قدرتها على مكافحة المرض بأمان.",
                "source": f"الملخص: \"{summary}\""
            },
            "en": {
                "simple": f"This publication (**{title}**) explores the rational design, synthesis, and biological profiling of therapeutic candidates, characterizing {num_c} distinct compounds across {num_b} bioactivity assays.",
                "facts": [
                    f"**Title:** {title}",
                    f"**Molecules Characterized:** {num_c} compounds",
                    f"**Bioactivity Screens:** {num_b} assay data points",
                    f"**Core Objective:** Structure-activity optimization for enhanced potency"
                ],
                "meaning": "It illustrates how systematic tweaks to a chemical scaffold can dramatically improve drug efficacy while minimizing side reactions.",
                "source": f"Executive Summary: \"{summary}\""
            }
        }
        return templates.get(lang, templates["en"])

    def _generate_general_chemistry_response(self, lang: str, query: str) -> Dict[str, Any]:
        """Provides general chemistry guidance when no paper is uploaded or for open inquiries."""
        templates = {
            "ml": {
                "simple": f"നിങ്ങൾ ചോദിച്ച **\"{query}\"** എന്ന ചോദ്യം പ്രധാനപ്പെട്ട ഒരു രസതന്ത്ര വിഷയമാണ്. കെമിസ്ട്രിയിൽ തന്മാത്രകളുടെ ഘടനയും അവയുടെ പ്രതിപ്രവർത്തനങ്ങളുമാണ് അടിസ്ഥാനം.",
                "facts": [
                    "**വിഷയം:** പൊതു രസതന്ത്ര വിശകലനം",
                    "**രീതി:** വ്യക്തമായ തന്മാത്രാ സമവാക്യങ്ങളും പരീക്ഷണ മാനദണ്ഡങ്ങളും",
                    "**ഉപയോഗം:** ഔഷധ നിർമ്മാണവും പരീക്ഷണശാലാ സിന്തസിസും"
                ],
                "meaning": "രസതന്ത്രം എന്നാൽ തന്മാത്രകളുടെ നിർമ്മാണവും മാറ്റവുമാണ്. ഇതിലൂടെയാണ് നാം കഴിക്കുന്ന മരുന്നുകളും ഉപയോഗിക്കുന്ന വസ്തുക്കളും നിർമ്മിക്കപ്പെടുന്നത്.",
                "source": "പൊതു രസതന്ത്ര ശാസ്ത്ര തത്വങ്ങൾ (General Chemical Knowledge)"
            },
            "hi": {
                "simple": f"आपका प्रश्न **\"{query}\"** रसायन विज्ञान का एक महत्वपूर्ण विषय है। अणुओं की संरचना और उनके बीच की रासायनिक प्रतिक्रियाएँ ही इसके मूल आधार हैं।",
                "facts": [
                    "**विषय:** सामान्य रासायनिक विश्लेषण",
                    "**आधार:** आणविक संरचना और प्रतिक्रिया गतिशीलता",
                    "**अनुप्रयोग:** औषधि निर्माण और अनुसंधान"
                ],
                "meaning": "रसायन विज्ञान हमें यह समझने में मदद करता है कि विभिन्न अणु एक-दूसरे के साथ कैसे जुड़ते हैं और नए उपयोगी पदार्थ बनाते हैं।",
                "source": "मानक रासायनिक संदर्भ (General Chemical Reference)"
            },
            "ta": {
                "simple": f"உங்கள் கேள்வி **\"{query}\"** வேதியியலில் மிக முக்கியமானதாகும். மூலக்கூறுகளின் கட்டமைப்பும் அவற்றின் செயல்பாடுகளுமே வேதியியலின் அடித்தளம்.",
                "facts": [
                    "**பொருள்:** பொது வேதியியல் கோட்பாடு",
                    "**அடிப்படை:** மூலக்கூறு கட்டமைப்பு மற்றும் வினைகள்",
                    "**பயன்பாடு:** மருந்து உற்பத்தி"
                ],
                "meaning": "மூலக்கூறுகள் எவ்வாறு இணைகின்றன என்பதைப் புரிந்து கொள்வது புதிய மருந்துகளை உருவாக்க இன்றியமையாதது.",
                "source": "பொது வேதியியல் அறிவு (General Chemistry Knowledge)"
            },
            "de": {
                "simple": f"Ihre Frage **\"{query}\"** betrifft ein zentrales chemisches Prinzip. Die Struktur und Reaktivität von Molekülen bilden das Fundament aller chemischen Prozesse.",
                "facts": [
                    "**Fachgebiet:** Allgemeine und organische Chemie",
                    "**Grundprinzip:** Struktur-Wirkungs-Beziehungen und Reaktionsmechanismen",
                    "**Bedeutung:** Pharmazeutische Forschung und Syntheseoptimierung"
                ],
                "meaning": "Das Verständnis molekularer Wechselwirkungen ermöglicht es Chemikern, gezielt Wirkstoffe mit gewünschten Eigenschaften zu entwerfen.",
                "source": "Allgemeines chemisches Lehrbuchwissen"
            },
            "es": {
                "simple": f"Su consulta **\"{query}\"** aborda un principio fundamental de la química. La reactividad y estructura de los átomos determinan las propiedades de la materia.",
                "facts": [
                    "**Área:** Química orgánica y medicinal",
                    "**Principio:** Mecanismos de reacción e interacciones moleculares",
                    "**Aplicación:** Síntesis y diseño de fármacos"
                ],
                "meaning": "Comprender cómo interactúan las moléculas permite diseñar compuestos útiles para la medicina y la industria.",
                "source": "Conocimiento químico fundamental"
            },
            "fr": {
                "simple": f"Votre question **\"{query}\"** touche à un concept clé de la chimie. La structure et la réactivité des molécules gouvernent l'ensemble des transformations chimiques.",
                "facts": [
                    "**Discipline :** Chimie organique et médicinale",
                    "**Principe :** Interactions moléculaires et mécanismes réactionnels",
                    "**Utilité :** Conception et synthèse de molécules actives"
                ],
                "meaning": "Comprendre les liaisons entre molécules est la clé pour fabriquer des médicaments ciblés et efficaces.",
                "source": "Principes généraux de chimie"
            },
            "zh": {
                "simple": f"您所咨询的问题 **\"{query}\"** 是化学领域的核心概念。分子结构与化学键反应机制决定了物质的所有基本特性。",
                "facts": [
                    "**学科范畴：** 有机化学与药物化学",
                    "**核心要素：** 分子构象、电荷分布与反应过渡态",
                    "**实际应用：** 药物发现与工艺合成开发"
                ],
                "meaning": "洞察分子间微观相互作用规律，是精准设计并高效合成新型治疗药物的基石。",
                "source": "通用化学专业基础原理"
            },
            "ar": {
                "simple": f"سؤالكم حول **\"{query}\"** يتناول مفهوماً كيميائياً أساسياً. إن التركيب الجزيئي والتفاعلات بين الذرات هي الركيزة لجميع العلوم الكيميائية.",
                "facts": [
                    "**المجال:** الكيمياء العضوية والدوائية",
                    "**الأساس:** الروابط الكيميائية وآليات التفاعل",
                    "**التطبيق:** تطوير الأدوية والتخليق المعملي"
                ],
                "meaning": "فهم كيفية تفاعل الجزيئات هو المفتاح لتصميم علاجات فعالة وآمنة.",
                "source": "المبادئ الكيميائية العامة"
            },
            "en": {
                "simple": f"Your inquiry regarding **\"{query}\"** addresses a fundamental chemical principle. Molecular architecture, bonding, and electronic properties govern all chemical behavior.",
                "facts": [
                    "**Domain:** Organic & Medicinal Chemistry",
                    "**Foundations:** Structure-activity dynamics and reaction pathways",
                    "**Application:** Drug discovery, process engineering, and chemical synthesis"
                ],
                "meaning": "Understanding how molecules interact enables chemists to deliberately design therapeutics that bind selectively to target receptors while avoiding unwanted toxicity.",
                "source": "Fundamental Chemistry Principles"
            }
        }
        return templates.get(lang, templates["en"])

    # -------------------------------------------------------------
    # Helper parsing utilities
    # -------------------------------------------------------------

    def _extract_section(self, text: str, header: str) -> str:
        """Extracts text following a markdown header until the next header."""
        escaped_h = re.escape(header.strip("# "))
        pattern = rf"###?\s*{escaped_h}\s*\n(.*?)(?=\n###?|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_bullet_points(self, text: str, header: str) -> List[str]:
        """Extracts bullet points from a section."""
        sec_text = self._extract_section(text, header)
        if not sec_text:
            return []
        bullets = []
        for line in sec_text.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.\s", line):
                clean_b = re.sub(r"^[-*]\s+|\d+\.\s+", "", line).strip()
                if clean_b:
                    bullets.append(clean_b)
        return bullets
