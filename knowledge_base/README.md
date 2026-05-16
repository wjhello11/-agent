# Clinical Knowledge Base

This knowledge base is split into three layers for the clinical nutrition agent.

## Layers

1. `structured/`
   Objective food facts for deterministic lookup. This layer should store food
   composition, GI/GL, purine values, allergen flags, and portion conversions.

2. `rag/` runtime database
   Evidence-based clinical guidance indexed from uploaded documents and legacy
   Markdown seed pages. This layer is stored in `data/clinical_rag.db` and used
   by `search_clinical_rag`.

3. `rules/`
   Hard safety rules for red-flag combinations, allergies, disease-food
   contraindications, and drug-food interactions.

## Operating Rule

Use the structured database for numbers, Clinical RAG for cited guidance, and
the rules layer for one-vote veto safety checks.
