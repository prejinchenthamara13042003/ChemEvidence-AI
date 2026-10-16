"""
Unit tests for MultilingualChemBot (core/multilingual_bot.py).
Tests:
- Automatic script-based and keyword-based language detection
- Manual language override
- Proper simple responses in Malayalam, Hindi, Tamil, German, Spanish, French, English
- 4-Tier structured layout (Simple Answer, Key Facts, Everyday Meaning, Grounding)
- Resilience with and without uploaded paper analysis
"""

import os
import unittest
from core.pdf_parser import DocumentParser
from core.evidence_engine import EvidenceEngine
from core.chemistry_extractor import ChemistryExtractor
from core.multilingual_bot import MultilingualChemBot


class TestMultilingualChemBot(unittest.TestCase):

    def setUp(self):
        self.bot = MultilingualChemBot(api_key=None, use_local_only=True)  # Use local multilingual engine

        sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_papers")
        txt_path = os.path.join(sample_dir, "paper1_kinase_inhibitors.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.pages = DocumentParser.parse_text_content(content)
        self.meta = DocumentParser.extract_metadata(self.pages)
        self.engine = EvidenceEngine(self.pages)
        extractor = ChemistryExtractor(use_local_only=True)
        self.analysis = extractor.analyze_paper(self.pages, self.meta, self.engine)

    def test_language_detection(self):
        # Malayalam script
        code, name = self.bot.detect_language("ഇതിലെ പ്രധാനപ്പെട്ട കോമ്പൗണ്ട് ഏതാണ്?")
        self.assertEqual(code, "ml")
        self.assertEqual(name, "Malayalam")

        # Explicit language request in prompt (e.g. English query asking for Spanish / Malayalam / Hindi)
        code_sp, name_sp = self.bot.detect_language("What is the lead molecule? Explain in Spanish please")
        self.assertEqual(code_sp, "es")
        self.assertEqual(name_sp, "Spanish")

        code_ml, name_ml = self.bot.detect_language("Can you explain the main compound in Malayalam?")
        self.assertEqual(code_ml, "ml")
        self.assertEqual(name_ml, "Malayalam")

        code_hi, name_hi = self.bot.detect_language("Tell me the yield in Hindi")
        self.assertEqual(code_hi, "hi")
        self.assertEqual(name_hi, "Hindi")

        # Hindi script
        code, name = self.bot.detect_language("इस पेपर का मुख्य निष्कर्ष क्या है?")
        self.assertEqual(code, "hi")
        self.assertEqual(name, "Hindi")

        # Tamil script
        code, name = self.bot.detect_language("இந்த ஆய்வின் முக்கிய முடிவு என்ன?")
        self.assertEqual(code, "ta")
        self.assertEqual(name, "Tamil")

        # Arabic script
        code, name = self.bot.detect_language("ما هو المركب الأكثر فعالية؟")
        self.assertEqual(code, "ar")
        self.assertEqual(name, "Arabic")

        # Chinese script
        code, name = self.bot.detect_language("这篇论文的核心结论是什么？")
        self.assertEqual(code, "zh")
        self.assertEqual(name, "Chinese")

        # German keywords
        code, name = self.bot.detect_language("Welche Reaktion lieferte die beste Ausbeute?")
        self.assertEqual(code, "de")
        self.assertEqual(name, "German")

        # Spanish keywords
        code, name = self.bot.detect_language("¿Cuál es el compuesto líder?")
        self.assertEqual(code, "es")
        self.assertEqual(name, "Spanish")

        # French keywords
        code, name = self.bot.detect_language("Quel est le rendement de cette réaction?")
        self.assertEqual(code, "fr")
        self.assertEqual(name, "French")

        # English
        code, name = self.bot.detect_language("What is the lead compound in this paper?")
        self.assertEqual(code, "en")
        self.assertEqual(name, "English")

    def test_manual_language_override(self):
        # English query with Malayalam manual override
        code, name = self.bot.detect_language("What is the yield?", user_choice="മലയാളം (Malayalam)")
        self.assertEqual(code, "ml")
        self.assertEqual(name, "Malayalam")

        # English query with German manual override
        code, name = self.bot.detect_language("Explain this", user_choice="Deutsch (German)")
        self.assertEqual(code, "de")
        self.assertEqual(name, "German")

    def test_malayalam_simple_answer_with_paper(self):
        query = "ഇതിലെ പ്രധാനപ്പെട്ട കോമ്പൗണ്ട് (lead compound) ഏതാണ്?"
        resp = self.bot.answer(
            query=query,
            analysis=self.analysis,
            pages=self.pages,
            language_choice="Auto-Detect",
            simplicity_level="Simple & Everyday"
        )
        self.assertEqual(resp.language_code, "ml")
        self.assertEqual(resp.detected_language, "Malayalam")
        self.assertIn("🎯 ലളിതമായ ഉത്തരം", resp.full_formatted_text)
        self.assertIn("🔬 പ്രധാന രാസവിവരങ്ങളും സംഖ്യകളും", resp.full_formatted_text)
        self.assertIn("💡 ഇത് എന്തുകൊണ്ട് പ്രധാനം?", resp.full_formatted_text)
        self.assertIn("📚 രേഖാമൂലമുള്ള തെളിവ്", resp.full_formatted_text)
        self.assertGreater(len(resp.key_facts), 0)

    def test_german_yield_answer(self):
        query = "Welche Reaktion lieferte die beste Ausbeute?"
        resp = self.bot.answer(
            query=query,
            analysis=self.analysis,
            pages=self.pages,
            language_choice="Auto-Detect",
            simplicity_level="Simple & Everyday"
        )
        self.assertEqual(resp.language_code, "de")
        self.assertIn("🎯 Einfache und direkte Antwort", resp.full_formatted_text)
        self.assertIn("🔬 Wichtige chemische Fakten", resp.full_formatted_text)
        self.assertIn("Ausbeute", resp.full_formatted_text)

    def test_ic50_concept_explanation_in_hindi(self):
        query = "IC50 क्या है?"
        resp = self.bot.answer(
            query=query,
            analysis=self.analysis,
            pages=self.pages,
            language_choice="Auto-Detect",
            simplicity_level="Simple & Everyday"
        )
        self.assertEqual(resp.language_code, "hi")
        self.assertIn("🎯 सरल उत्तर", resp.full_formatted_text)
        self.assertIn("50%", resp.full_formatted_text)
        self.assertTrue(
            any(w in resp.full_formatted_text for w in ["कम खुराक में अधिक असर", "असरदार", "शक्तिशाली", "दवा"]),
            f"Expected Hindi potency concept in response: {resp.full_formatted_text}"
        )

    def test_open_chemistry_mode_without_paper(self):
        # When no paper is loaded
        query = "What is a kinase inhibitor?"
        resp = self.bot.answer(
            query=query,
            analysis=None,
            pages=None,
            language_choice="Auto-Detect",
            simplicity_level="Simple & Everyday"
        )
        self.assertEqual(resp.language_code, "en")
        self.assertIsNotNone(resp.simple_answer)
        self.assertGreater(len(resp.key_facts), 0)
        self.assertIn("🎯 Simple & Direct Answer", resp.full_formatted_text)

    def test_answer_with_engine_and_chat_history(self):
        """Verifies that ChemBot runs with EvidenceEngine and chat history without errors."""
        history = [
            {"question": "What is EGFR?", "simple_answer": "EGFR is an epidermal growth factor receptor protein."}
        ]
        resp = self.bot.answer(
            query="What is the IC50 of compound 3b?",
            analysis=self.analysis,
            pages=self.pages,
            engine=self.engine,
            chat_history=history,
            language_choice="English",
            simplicity_level="Simple & Everyday"
        )
        self.assertEqual(resp.language_code, "en")
        self.assertIsNotNone(resp.simple_answer)
        self.assertGreater(len(resp.key_facts), 0)
        self.assertTrue("3b" in resp.full_formatted_text.lower() or "8.4" in resp.full_formatted_text)

    def test_specific_compound_matching_in_bot(self):
        """Verifies that asking about compound 3b retrieves compound 3b rather than only the global lead."""
        resp = self.bot.answer(
            query="Tell me about compound 3b",
            analysis=self.analysis,
            pages=self.pages,
            engine=self.engine,
            language_choice="English"
        )
        self.assertIn("3b", resp.full_formatted_text.lower())


if __name__ == "__main__":
    unittest.main()
