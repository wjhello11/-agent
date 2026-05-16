# Clinical Nutrition LLMWiki

This folder follows an `llmwiki`-style layout for the nutrition agent.

- `raw/` keeps immutable source material notes and citations.
- topic folders keep synthesized wiki articles for retrieval.
- `_index.md` is the human-maintained navigation page.

The runtime plugin `search_from_llmwiki` searches the synthesized Markdown
articles first and skips `raw/` by default.
