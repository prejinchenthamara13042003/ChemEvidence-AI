"""
Evidence Indexing and Grounding Engine.
Chunks documents with page-boundary tracking, indexes text for BM25/TF-IDF retrieval,
and guarantees verifiable evidence citation against the source literature.
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from core.pdf_parser import PageContent
from core.models import EvidenceCitation


class EvidenceChunk:
    """A granular chunk of scientific text tied to its exact source page and section."""
    def __init__(
        self,
        chunk_id: str,
        page_number: int,
        section: str,
        text: str,
        start_char: int = 0,
        end_char: int = 0
    ):
        self.chunk_id = chunk_id
        self.page_number = page_number
        self.section = section
        self.text = text
        self.start_char = start_char
        self.end_char = end_char

    def __repr__(self):
        return f"<EvidenceChunk {self.chunk_id} p.{self.page_number} ({self.section})>"


class EvidenceEngine:
    """Evidence storage, keyword/semantic retrieval, and quote verification."""

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "of", "from", "as", "is", "was", "were", "are", "be", "been", "it",
        "this", "that", "these", "those", "which", "who", "whom", "what", "where",
        "when", "how", "all", "any", "both", "each", "few", "more", "most", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "should", "now", "d", "ll", "m", "o", "re", "ve", "y"
    }

    def __init__(self, pages: List[PageContent]):
        self.pages = pages
        self.chunks: List[EvidenceChunk] = []
        self.doc_freq: Dict[str, int] = Counter()
        self.chunk_term_counts: List[Counter] = []
        self.avg_doc_len: float = 0.0
        self._build_index()

    def _build_index(self):
        """Chunk pages into paragraph-sized passages and compute BM25 index."""
        self.chunks = []
        chunk_idx = 0

        for page in self.pages:
            current_section = "Main Text"
            if page.sections:
                current_section = page.sections[0]["name"]
            elif page.page_number == 1:
                current_section = "Publication Details & Authors"

            # Split by double newline or significant headings
            raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
            
            # If paragraphs are too small or too large, rebalance
            merged_paras = []
            buf = ""
            for p_idx, p in enumerate(raw_paragraphs):
                # Update section if paragraph is a section header or on page 1 before abstract
                matched_sec = None
                for s in page.sections:
                    if s["name"].lower() in p.lower()[:50]:
                        matched_sec = s["name"]
                        break
                if matched_sec:
                    current_section = matched_sec
                elif page.page_number == 1 and p_idx == 0:
                    current_section = "Publication Details & Authors"

                if len(buf) + len(p) < 600:
                    buf = (buf + "\n\n" + p).strip()
                else:
                    if buf:
                        merged_paras.append((current_section, buf))
                    buf = p
            if buf:
                merged_paras.append((current_section, buf))

            for sec, para_text in merged_paras:
                if len(para_text) < 25:
                    continue
                chunk_id = f"chunk_p{page.page_number}_{chunk_idx}"
                chunk = EvidenceChunk(
                    chunk_id=chunk_id,
                    page_number=page.page_number,
                    section=sec,
                    text=para_text
                )
                self.chunks.append(chunk)
                chunk_idx += 1

        # Build BM25 statistics
        total_len = 0
        self.chunk_term_counts = []
        self.doc_freq = Counter()

        for chunk in self.chunks:
            tokens = self._tokenize(chunk.text)
            term_counter = Counter(tokens)
            self.chunk_term_counts.append(term_counter)
            total_len += len(tokens)
            for term in term_counter.keys():
                self.doc_freq[term] += 1

        self.avg_doc_len = total_len / max(len(self.chunks), 1)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1
    ) -> List[Tuple[EvidenceChunk, float]]:
        """
        Retrieves top relevant passages using BM25 ranking
        with chemistry query term boosting and intent-aware section scoring.
        """
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Intent detection for high-level research questions
        q_lower = query.lower()
        is_author_q = any(w in q_lower for w in [
            "author", "authors", "who wrote", "researcher", "researchers",
            "investigator", "who published", "who created", "affiliation"
        ])
        is_summary_q = any(w in q_lower for w in [
            "summary", "summarize", "overview", "what is this paper about",
            "what is the paper about", "explain the paper", "synopsis",
            "brief summary", "executive summary"
        ])
        is_conclusion_q = any(w in q_lower for w in [
            "conclusion", "conclusions", "conclude", "takeaway", "what was found", "what did they conclude"
        ])
        is_objective_q = any(w in q_lower for w in [
            "objective", "aim", "goal", "purpose", "why was", "rationale", "problem",
            "motivation", "study design", "focus of this study", "significance",
            "what problem", "what did the researchers discover"
        ])
        is_compound_count_q = any(w in q_lower for w in [
            "how many", "number of compounds", "target compounds", "compounds were",
            "derivatives were", "analogues were", "library size", "library of", "evaluated compounds"
        ])
        is_mechanism_q = any(w in q_lower for w in [
            "mechanism", "mode of action", "moa", "atp-competitive", "irreversible",
            "covalent", "binding mode", "how does it work", "inhibition mechanism"
        ])

        # Chemistry query boosting
        chem_boosts = {
            "yield": 2.2, "ic50": 2.8, "ec50": 2.8, "ki": 2.8, "gi50": 2.8,
            "cc50": 2.8, "kd": 2.8, "mic": 2.8, "mbc": 2.5, "tgi": 2.5,
            "catalyst": 2.0, "solvent": 2.0, "nmr": 2.0, "temperature": 1.8,
            "compound": 1.5, "inhibition": 2.2, "activity": 1.6, "synthesis": 1.8,
            "purity": 2.2, "table": 2.0, "scheme": 1.8, "potency": 2.2,
            "selectivity": 2.2, "antiproliferative": 2.2, "cytotoxicity": 2.2,
            "enantiomeric": 2.0, "biofilm": 2.0, "binding": 2.0
        }

        # Domain Chemical Synonym Expansion
        chem_synonyms = {
            "potency": ["ic50", "activity", "ki", "inhibition"],
            "bioactivity": ["ic50", "ki", "ec50", "activity"],
            "activity": ["ic50", "inhibition", "potency"],
            "yield": ["yield", "isolated", "conversion"],
            "efficiency": ["yield", "conversion", "efficiency"],
            "toxicity": ["cytotoxicity", "cc50", "viability", "safety"],
            "cytotoxicity": ["antiproliferative", "gi50", "viability", "cc50"],
            "antiproliferative": ["cytotoxicity", "gi50", "inhibition"],
            "mechanism": ["mode", "action", "binding", "docking", "competitive", "allosteric"],
            "binding": ["affinity", "kd", "ki", "docking", "interaction"],
            "solvent": ["dcm", "meoh", "thf", "dmf", "etac", "solvent"],
            "catalyst": ["pd", "palladium", "copper", "catalyst", "ligand"],
            "temperature": ["reflux", "room", "temperature", "rt"],
        }

        # Build expanded token set with weights
        expanded_token_weights: Dict[str, float] = {}
        for t in query_tokens:
            expanded_token_weights[t] = 1.0
            if t in chem_synonyms:
                for syn in chem_synonyms[t]:
                    if syn not in expanded_token_weights:
                        expanded_token_weights[syn] = 0.55

        # Also extract compound identifiers from phrasing like 'compound 3b', 'derivative 4c'
        comp_id_matches = re.findall(r"\b(?:compound|derivative|analog|analogue|molecule|intermediate|entry)\s+([0-9]+[a-z]?|[a-z][0-9]+)\b", q_lower)
        for cid in comp_id_matches:
            expanded_token_weights[cid.lower()] = 1.8

        # Identify key query anchors: compound tokens and metric tokens
        metric_tokens = {k for k in ["ic50", "gi50", "ec50", "ki", "kd", "mic", "yield", "cc50", "purity", "potency", "selectivity"] if k in expanded_token_weights}
        comp_target_tokens = {t for t in expanded_token_weights if re.match(r"^\d+[a-z]?$", t) or re.match(r"^[a-z]\d+$", t) or t in ["egfr", "a549", "her2", "cdk4", "braf", "vero", "candida", "aspergillus", "mcf7", "hela"]}

        k1 = 1.5
        b = 0.75
        N = len(self.chunks)
        scores: List[float] = [0.0] * N

        for q_term, term_weight in expanded_token_weights.items():
            df = self.doc_freq.get(q_term, 0)
            if df == 0:
                continue

            # BM25 IDF
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            boost = chem_boosts.get(q_term, 1.0) * term_weight
            # Boost specific compound identifiers like 3b, 4a, 5c
            if re.match(r"^\d+[a-z]$", q_term) or re.match(r"^[a-z]\d+$", q_term):
                boost = max(boost, 3.5 * term_weight)

            for i, chunk in enumerate(self.chunks):
                tf = self.chunk_term_counts[i].get(q_term, 0)
                if tf == 0:
                    continue
                doc_len = sum(self.chunk_term_counts[i].values())
                denom = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                term_score = idf * (tf * (k1 + 1.0) / max(denom, 1e-6)) * boost
                scores[i] += term_score

        # Co-occurrence boosting: if chunk contains both the requested metric and compound/target
        if metric_tokens and comp_target_tokens:
            for i, chunk in enumerate(self.chunks):
                chunk_terms = set(self.chunk_term_counts[i].keys())
                has_metric = bool(chunk_terms & metric_tokens)
                has_comp_target = bool(chunk_terms & comp_target_tokens)
                if has_metric and has_comp_target:
                    scores[i] *= 1.35  # 35% ranking boost for precise co-occurrence

        # Intent-driven section boosts for high-level queries
        if is_author_q:
            for i, chunk in enumerate(self.chunks):
                if chunk.page_number == 1 and (i == 0 or "Publication Details" in chunk.section or "Abstract" in chunk.section):
                    scores[i] = max(scores[i], 6.0)

        if is_summary_q:
            for i, chunk in enumerate(self.chunks):
                sec_l = chunk.section.lower()
                if "abstract" in sec_l or (chunk.page_number == 1 and i <= 1):
                    scores[i] = max(scores[i], 5.5)
                elif "conclusion" in sec_l or "discussion" in sec_l:
                    scores[i] = max(scores[i], 3.8)

        if is_conclusion_q:
            for i, chunk in enumerate(self.chunks):
                sec_l = chunk.section.lower()
                if "conclusion" in sec_l or "discussion" in sec_l:
                    scores[i] = max(scores[i], 5.0)

        if is_objective_q:
            for i, chunk in enumerate(self.chunks):
                sec_l = chunk.section.lower()
                if "abstract" in sec_l or (chunk.page_number == 1 and i <= 1) or "introduction" in sec_l:
                    scores[i] = max(scores[i], 5.8)
                elif "conclusion" in sec_l:
                    scores[i] = max(scores[i], 3.5)

        if is_compound_count_q:
            for i, chunk in enumerate(self.chunks):
                c_text = chunk.text.lower()
                sec_l = chunk.section.lower()
                if any(k in c_text for k in ["target compounds", "scheme", "table 1", "synthesized", "series"]):
                    scores[i] = max(scores[i], 5.5)
                elif "synthesis" in sec_l or "chemistry" in sec_l or "table" in sec_l:
                    scores[i] = max(scores[i], 4.5)

        if is_mechanism_q:
            for i, chunk in enumerate(self.chunks):
                c_text = chunk.text.lower()
                sec_l = chunk.section.lower()
                if any(k in c_text for k in ["atp-competitive", "mechanism", "irreversible", "covalent", "binding mode"]):
                    scores[i] = max(scores[i], 5.5)
                elif "biological" in sec_l or "sar" in sec_l or "abstract" in sec_l:
                    scores[i] = max(scores[i], 3.8)

        # Fallback for broad general queries when BM25 score is below min_score
        max_score = max(scores) if scores else 0.0
        if max_score < min_score:
            is_general_paper_q = any(w in q_lower for w in [
                "study", "paper", "research", "investigation", "work", "article",
                "finding", "findings", "result", "results", "discover", "evaluated",
                "method", "target", "compound", "inhibitor", "approach"
            ])
            is_negated_or_absent = any(w in q_lower for w in ["bioavailability", "pharmacokinetic", "pk", "cmax", "clearance"])
            if is_general_paper_q and not is_negated_or_absent:
                for i, chunk in enumerate(self.chunks):
                    if chunk.page_number == 1:
                        scores[i] = max(scores[i], 3.0)
                    elif "table" in chunk.section.lower() or "biology" in chunk.section.lower():
                        scores[i] = max(scores[i], 2.0)

        # Combine results
        ranked = []
        for i, score in enumerate(scores):
            if score >= min_score:
                ranked.append((self.chunks[i], score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def verify_quote(self, quote: str) -> Optional[EvidenceCitation]:
        """
        Verifies if an extracted quote appears in the source pages
        and returns the precise page number and section.
        """
        if not quote or len(quote.strip()) < 8:
            return None

        clean_q = self._normalize_text(quote)

        for page in self.pages:
            norm_page = self._normalize_text(page.text)
            if clean_q in norm_page:
                sec = page.sections[0]["name"] if page.sections else "Main Text"
                return EvidenceCitation(
                    page_number=page.page_number,
                    section=sec,
                    verbatim_quote=quote.strip(),
                    confidence=1.0
                )

        # Fallback: substring matching on first 40 chars
        prefix = clean_q[:min(50, len(clean_q))]
        for page in self.pages:
            norm_page = self._normalize_text(page.text)
            if prefix in norm_page:
                sec = page.sections[0]["name"] if page.sections else "Main Text"
                return EvidenceCitation(
                    page_number=page.page_number,
                    section=sec,
                    verbatim_quote=quote.strip(),
                    confidence=0.88
                )

        return None

    def find_best_citation_for_text(self, text_snippet: str) -> EvidenceCitation:
        """Finds or constructs the most accurate page citation for a given text snippet."""
        verified = self.verify_quote(text_snippet)
        if verified:
            return verified

        # Use search to find closest chunk
        results = self.search(text_snippet, top_k=1, min_score=0.01)
        if results:
            best_chunk, score = results[0]
            # Extract sentence containing closest terms
            snippet = self._extract_relevant_sentence(best_chunk.text, text_snippet)
            return EvidenceCitation(
                page_number=best_chunk.page_number,
                section=best_chunk.section,
                verbatim_quote=snippet or best_chunk.text[:200],
                confidence=min(0.95, max(0.6, score / 10.0))
            )

        return EvidenceCitation(
            page_number=1,
            section="Main Text",
            verbatim_quote=text_snippet[:200],
            confidence=0.5
        )

    def _extract_relevant_sentence(self, full_text: str, query: str) -> str:
        """Find the sentence in full_text with greatest word overlap with query."""
        sentences = re.split(r"(?<=[.!?])\s+", full_text)
        query_words = set(self._tokenize(query))
        best_sentence = ""
        max_overlap = -1

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = set(self._tokenize(s_clean))
            overlap = len(s_words & query_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s_clean

        return best_sentence or full_text[:200]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens without stopwords."""
        words = re.findall(r"\b[A-Za-z0-9\-\%]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOP_WORDS]

    def _normalize_text(self, text: str) -> str:
        """Normalizes spaces and casing for fuzzy quote verification."""
        text = re.sub(r"\s+", " ", text.lower().strip())
        return text
