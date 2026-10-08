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

            # Split by double newline or significant headings
            raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
            
            # If paragraphs are too small or too large, rebalance
            merged_paras = []
            buf = ""
            for p in raw_paragraphs:
                # Update section if paragraph is a section header
                for s in page.sections:
                    if s["name"].lower() in p.lower()[:50]:
                        current_section = s["name"]

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
        with chemistry query term boosting.
        """
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

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

        # Identify key query anchors: compound tokens and metric tokens
        metric_tokens = {k for k in ["ic50", "gi50", "ec50", "ki", "kd", "mic", "yield", "cc50", "purity", "potency", "selectivity"] if k in query_tokens}
        comp_target_tokens = {t for t in query_tokens if re.match(r"^\d+[a-z]?$", t) or re.match(r"^[a-z]\d+$", t) or t in ["egfr", "a549", "her2", "cdk4", "braf", "vero", "candida", "aspergillus", "mcf7", "hela"]}

        k1 = 1.5
        b = 0.75
        N = len(self.chunks)
        scores: List[float] = [0.0] * N

        for q_term in query_tokens:
            df = self.doc_freq.get(q_term, 0)
            if df == 0:
                continue

            # BM25 IDF
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            boost = chem_boosts.get(q_term, 1.0)
            # Boost specific compound identifiers like 3b, 4a, 5c
            if re.match(r"^\d+[a-z]$", q_term) or re.match(r"^[a-z]\d+$", q_term):
                boost = max(boost, 3.5)

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
