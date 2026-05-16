# Clinical Safety Rules

This layer stores hard safety rules for the clinical nutrition agent.

Rules are designed for deterministic checks before the LLM produces advice.
The intended flow is:

1. Normalize user foods, allergies, diseases, medications, and risk factors.
2. Evaluate `clinical_safety_rules.json`.
3. If a `block` rule fires, return a red-flag response or require clinician
   confirmation before continuing.

The first rule set covers high-priority drug-food interactions, allergies, and
metabolic disease contraindications.
