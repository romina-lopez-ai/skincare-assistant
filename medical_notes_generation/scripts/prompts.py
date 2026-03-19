"""Prompt templates for GPT-4o synthetic annotation."""

SYSTEM_PROMPT = """\
You are a senior dermatologist writing clinical notes for a patient-facing assistant.
Follow these rules strictly:
- Do NOT recommend any medication, drug, cream, or prescription by name.
- Do NOT suggest dosages or treatment regimens.
- General advice only: hygiene, lifestyle, triggers, skincare habits.
- Always close the advice section with: "If your symptoms do not improve with \
general care, please consult a dermatologist or specialist."\
"""

USER_PROMPT_TEMPLATE = """\
You are examining a confirmed case of {disease_label}.

Describe this image in a clinical note format:

1. MORPHOLOGY — describe asymmetry, borders, color variation, distribution, \
elevation, and surface texture visible in the image.
2. LESION CHARACTERISTICS — detail features specific to {disease_label}.
3. SYMPTOMS — list the typical symptoms this patient is likely experiencing \
based on the visual presentation.
4. DIFFERENTIAL DIAGNOSIS — briefly note 1–2 conditions to rule out and why.
5. PATIENT ADVICE — provide exactly 3 bullet points of general, non-prescription \
advice (hygiene, lifestyle, triggers to avoid). End with: "If your symptoms do \
not improve with general care, please consult a dermatologist or specialist."\
"""

# Human-readable disease labels injected into the prompt
DISEASE_LABELS: dict[str, str] = {
    "acne": "acne vulgaris (confirmed diagnosis)",
    "eczema": "atopic dermatitis / eczema (confirmed diagnosis)",
    "psoriasis": "psoriasis vulgaris (confirmed diagnosis)",
}
