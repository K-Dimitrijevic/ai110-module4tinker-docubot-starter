"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import string

# Common words that carry no topic meaning. They are ignored when scoring,
# so a question can't "match" a section just by sharing words like "how" or "the".
STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "me", "mention",
    "my", "of", "on", "or", "the", "there", "these", "this", "to", "what",
    "when", "where", "which", "who", "why", "with", "docs",
}

# Refusal guardrail: a section must contain at least this many of the
# question's meaningful (non-stopword) words to count as evidence. Below that,
# retrieve() returns nothing and the answer methods say "I do not know".
MIN_SCORE = 1

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split documents into heading-based sections
        self.sections = self.split_into_sections(self.documents)  # List of (filename, section_text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.sections)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    def split_into_sections(self, documents):
        """
        Splits each document into sections at Markdown headings.
        Each section is a heading plus everything under it, up to the next heading.
        Returns a list of tuples: (filename, section_text)
        """
        # Called once from __init__ with the output of load_documents().
        # The result is stored in self.sections, which build_index() and
        # retrieve() search instead of whole files.
        sections = []

        for filename, text in documents:
            current = []        # Lines of the section being built
            in_code_block = False

            for line in text.splitlines():
                # Track ``` fences so a "#" comment inside a code block
                # is not mistaken for a heading.
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block

                # A line starting with "#" (outside code) begins a new section,
                # so save the lines collected so far as a finished section.
                if line.startswith("#") and not in_code_block and current:
                    sections.append((filename, "\n".join(current).strip()))
                    current = []

                current.append(line)

            # Save the last section of the file (there's no heading after it).
            if current:
                sections.append((filename, "\n".join(current).strip()))

        # Drop sections that ended up empty (e.g. only blank lines).
        return [(filename, text) for filename, text in sections if text]

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        # Called once from __init__ with the (filename, section_text) list
        # produced by split_into_sections(). The result is stored in self.index
        # and later used by retrieve() to quickly find which sections mention
        # a query word. Each word maps to section positions (0, 1, 2, ...) in
        # self.sections, since one file now has many sections.
        index = {}

        # Walk every section, and every whitespace-separated word inside it.
        # enumerate() gives each section's position alongside its contents.
        for position, (_, text) in enumerate(documents):
            for word in text.split():
                # Normalize the word: lowercase and strip surrounding punctuation,
                # so "Token," and "token" map to the same key. score_document()
                # and retrieve() normalize words the same way so lookups match.
                token = word.lower().strip(string.punctuation)

                # Skip words that were pure punctuation (e.g. "---", "#").
                if not token:
                    continue

                # Create an empty section list the first time we see this word.
                if token not in index:
                    index[token] = []

                # Record this section under the word, only once per section.
                if position not in index[token]:
                    index[token].append(position)

        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        # Called by retrieve() for each candidate document to rank it.

        # Normalize the document's words (same rules as build_index) and put
        # them in a set for fast "is this word present?" checks.
        text_words = set(
            word.lower().strip(string.punctuation) for word in text.split()
        )

        # Count how many meaningful query words appear in the document.
        # Stopwords are skipped, and each distinct word counts once.
        score = 0
        for token in self.meaningful_words(query):
            if token in text_words:
                score += 1

        # Higher score = more query words matched = more relevant.
        return score

    def meaningful_words(self, text):
        """
        Returns the set of normalized words in text, minus STOPWORDS.
        """
        # Used by score_document() and retrieve() so both agree on which
        # query words count as evidence.
        words = set()
        for word in text.split():
            token = word.lower().strip(string.punctuation)
            if token and token not in STOPWORDS:
                words.add(token)
        return words

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        # Used by answer_retrieval_only() (Phase 1), answer_rag() (Phase 2,
        # which passes the results to llm_client.answer_from_snippets), and
        # evaluation.py (which checks the returned filenames for hit rate).
        results = []

        # Guardrail: if the question has no meaningful words (e.g. "how do i?"),
        # there is nothing to look for, so refuse right away.
        query_words = self.meaningful_words(query)
        if not query_words:
            return []

        # Step 1: use self.index (built by build_index) to collect the
        # positions of sections that contain at least one meaningful query
        # word. Unknown words return [].
        candidates = set()
        for token in query_words:
            candidates.update(self.index.get(token, []))

        # Step 2: score only the candidate sections with score_document().
        # self.sections comes from split_into_sections(). Guardrail: sections
        # below MIN_SCORE are dropped, so matches on filler words alone never
        # reach the answer methods or the LLM.
        for position in candidates:
            filename, text = self.sections[position]
            score = self.score_document(query, text)
            if score >= MIN_SCORE:
                results.append((score, filename, text))

        # Step 3: sort best matches first, then drop the score so callers get
        # the (filename, text) pairs they expect.
        results.sort(key=lambda item: item[0], reverse=True)
        results = [(filename, text) for _, filename, text in results]

        # Step 4: keep only the top_k results. An empty list (no meaningful
        # evidence) makes the answer methods reply "I do not know based on
        # these docs." instead of guessing.
        return results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
