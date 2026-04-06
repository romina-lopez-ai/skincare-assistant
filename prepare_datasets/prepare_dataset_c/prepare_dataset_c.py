"""Dataset C — QA-RAG with Reasoning.

Teaches the model to:
1. Receive context in USER role with PRIORITY tags.
2. Reason inside <think></think> tags before answering.
3. Compare multiple sources and prioritize by PRIORITY level.
4. Cite the source in the final answer using varied phrasing.
5. Handle 8 types of edge cases honestly without hallucinating.
6. Never prescribe or give specific medical treatments.

Edge case types:
  - irrelevant_domain      : context is from a completely different specialty
  - no_context             : retriever returned nothing (empty context block)
  - too_generic            : context mentions the topic but too vaguely to answer
  - partial_overlap        : context covers related concept but not the specific one asked
  - multi_condition_gap    : user asks about two conditions, context only covers one
  - out_of_scope_question  : user asks something outside dermatology
  - medication_not_in_ctx  : user asks about a specific drug not mentioned in context
  - second_opinion         : user wants model to override their doctor's advice

Provider : GROQ  — Llama 3.3 70B

Usage:
    pip install groq python-dotenv
    python dataset_c_groq.py --num_examples 300
"""

import json, os, re, time, logging, random, argparse, hashlib, sys
from groq import Groq
from dotenv import load_dotenv
from collections import defaultdict
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "prepare_datasets"))
from system_prompt import SYSTEM_UNIFIED

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT   = PROJECT_ROOT

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# TOPICS & FACTS
# ─────────────────────────────────────────────────────────────────────────────
COMPLIANT_TOPICS = [
    {
        "topic": "psoriasis",
        "facts": [
            "Psoriasis is a chronic condition where skin cells build up rapidly, forming thick scales.",
            "Common triggers include stress, infections, cold weather, and certain medications.",
            "A dermatologist may recommend topical therapies or other treatments based on your specific case.",
            "Keeping skin moisturized and avoiding known triggers can help manage symptoms.",
            "Psoriasis affects about 2-3% of the world population and has a genetic component."
        ]
    },
    {
        "topic": "acne",
        "facts": [
            "Acne occurs when hair follicles become clogged with oil and dead skin cells.",
            "Factors like hormones, stress, and certain cosmetics can influence breakouts.",
            "A dermatologist can help determine the right approach, which may include topical options.",
            "Gentle cleansing and avoiding picking at lesions can reduce irritation.",
            "Diet and lifestyle factors may play a role in acne severity for some individuals."
        ]
    },
    {
        "topic": "eczema",
        "facts": [
            "Eczema (atopic dermatitis) causes dry, itchy, and inflamed skin.",
            "Flare-ups can be triggered by irritants, allergens, stress, or weather changes.",
            "A dermatologist may suggest moisturizers, lifestyle adjustments, or prescription options if needed.",
            "Using gentle skincare products and avoiding harsh soaps can help.",
            "Eczema often begins in childhood and may persist into adulthood."
        ]
    },
    {
        "topic": "rosacea",
        "facts": [
            "Rosacea causes facial redness and sometimes small, red, pus-filled bumps.",
            "Triggers include sun exposure, spicy foods, alcohol, and stress.",
            "Unlike acne, rosacea typically does not feature blackheads (comedones).",
            "Sun protection and gentle skincare are essential for management.",
            "It is more common in fair-skinned adults over the age of 30."
        ]
    },
    {
        "topic": "seborrheic dermatitis",
        "facts": [
            "Seborrheic dermatitis causes scaly patches, red skin, and stubborn dandruff.",
            "It commonly affects oily areas like the scalp, sides of the nose, and eyebrows.",
            "It is often confused with psoriasis but scales are usually yellowish and greasy.",
            "Stress and cold, dry weather can trigger flare-ups.",
            "A dermatologist can recommend specific medicated shampoos or creams."
        ]
    },
    {
        "topic": "contact dermatitis",
        "facts": [
            "Contact dermatitis is a red, itchy rash caused by direct contact with a substance.",
            "Common triggers include soaps, cosmetics, fragrances, jewelry (nickel), and plants.",
            "The rash usually appears minutes to hours after exposure and can last 2-4 weeks.",
            "Identifying and avoiding the trigger is the key to preventing recurrence.",
            "Cool compresses and moisturizers can help soothe the affected area."
        ]
    },
    {
        "topic": "urticaria",
        "facts": [
            "Urticaria (hives) are itchy, raised welts that are skin-colored or red.",
            "They appear suddenly and can be triggered by allergies, stress, or infections.",
            "Hives typically blanch (turn white) when pressed in the center.",
            "Most cases are acute and resolve quickly, but some can become chronic.",
            "If hives are accompanied by difficulty breathing, seek emergency care immediately."
        ]
    },
    {
        "topic": "tinea corporis",
        "facts": [
            "Tinea corporis (ringworm) is a fungal infection that often looks like a red ring.",
            "The center of the ring may be clear, while the edges are scaly and raised.",
            "It is contagious and spreads through skin-to-skin contact or shared items.",
            "A dermatologist can confirm the fungal origin and suggest antifungal options.",
            "Keeping the skin clean and dry is fundamental for recovery."
        ]
    },
    {
        "topic": "actinic keratosis",
        "facts": [
            "Actinic keratoses are rough, scaly patches on the skin caused by years of sun exposure.",
            "They are most common on the face, scalp, and back of the hands.",
            "They are considered precancerous and can develop into squamous cell carcinoma.",
            "A dermatologist should evaluate these patches and may recommend removal.",
            "Strict sun protection is mandatory for patients with these lesions."
        ]
    },
    {
        "topic": "vitiligo",
        "facts": [
            "Vitiligo is a condition where skin loses its pigment cells (melanocytes).",
            "This results in discolored patches in different areas of the body, including hair.",
            "The exact cause is unknown, but it is considered an autoimmune condition.",
            "It is not contagious and cannot be caught from someone else.",
            "A dermatologist can offer treatments to help restore some skin tone or even it out."
        ]
    },
    {
        "topic": "cellulitis",
        "facts": [
            "Cellulitis is a deep bacterial skin infection that causes redness, swelling, and pain.",
            "The affected skin usually feels hot and very tender to the touch.",
            "It can spread rapidly and become life-threatening if not treated with antibiotics.",
            "Symptoms like fever, chills, and red streaks indicate a serious infection.",
            "Seek immediate medical attention if you suspect you have cellulitis."
        ]
    },
    {
        "topic": "perioral dermatitis",
        "facts": [
            "Perioral dermatitis causes small, red, pus-filled bumps around the mouth and nose.",
            "It is often mistaken for acne, but it typically lacks blackheads or whiteheads.",
            "Using strong topical steroids or heavy face creams can trigger or worsen it.",
            "A dermatologist may recommend stopping certain products and using gentle alternatives.",
            "It is most common in young to middle-aged women and can be itchy or burning."
        ]
    },
    {
        "topic": "lichen planus",
        "facts": [
            "Lichen planus is an inflammatory condition that causes shiny, flat-topped, purplish bumps.",
            "It commonly appears on the inner wrists, forearms, and ankles.",
            "It is often mistaken for psoriasis, but the bumps are usually purple rather than bright red.",
            "It is extremely itchy and is not contagious.",
            "A dermatologist can confirm it through a skin biopsy and suggest management options."
        ]
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TYPES
# ─────────────────────────────────────────────────────────────────────────────
QUESTION_TYPES = {
    "definition":  [
        "What is {topic}?",
        "Can you explain {topic}?",
        "What exactly is {topic}?",
    ],
    "cause": [
        "What causes {topic}?",
        "Why do people develop {topic}?",
        "What triggers {topic}?",
    ],
    "management": [
        "How can I manage {topic}?",
        "What helps with {topic} symptoms?",
        "What should I do about {topic}?",
    ],
    "doctor": [
        "Should I see a doctor for {topic}?",
        "When should I see a dermatologist about {topic}?",
        "Is {topic} serious enough to require medical attention?",
    ],
    "duration": [
        "How long does {topic} last?",
        "Will {topic} go away on its own?",
        "Is {topic} a chronic condition?",
    ],
}

GENERAL_FOLLOWUPS = [
    "Do you want to know more about {topic}?",
    "Do you have any other questions?",
    "Is there anything else I can help you with?",
    "Would you like information about any other skin condition?",
    "Feel free to ask if you have more questions about {topic}.",
    "Let me know if you'd like to explore this topic further.",
]

# ─────────────────────────────────────────────────────────────────────────────
# CITATION PHRASES — varied so the model doesn't always repeat the same wording
# ─────────────────────────────────────────────────────────────────────────────
CITATION_PHRASES = [
    "According to {source}",
    "Based on {source}",
    "As noted in {source}",
    "The {source} indicates that",
    "Per {source}",
    "{source} states that",
    "Drawing from {source}",
    "As reported by {source}",
    "Referring to {source}",
    "In line with {source}",
]

# ─────────────────────────────────────────────────────────────────────────────
# SOURCES
# ─────────────────────────────────────────────────────────────────────────────
SOURCES_PRIMARY = [
    "Dermatology Knowledge Base — Clinical Study",
    "AAD Clinical Practice Guidelines",
    "British Association of Dermatologists Evidence Review",
    "PubMed Systematic Review (2023)",
    "UpToDate Dermatology Module",
    "Journal of Investigative Dermatology — Meta-analysis",
    "WHO Skin Health Resource",
    "Mayo Clinic Dermatology Reference",
]

SOURCES_SECONDARY = [
    "General Health Guidelines",
    "Patient Education Resource — NHS",
    "MedlinePlus Skin Health Overview",
    "WebMD Medical Review Board",
    "CDC Skin Conditions Resource",
    "National Eczema Association Guide",
    "Skin Cancer Foundation Recommendations",
]

SOURCES_IRRELEVANT = [
    "Cardiology Notes — Cardiac Care Manual",
    "Orthopedic Surgery Textbook",
    "Nephrology Clinical Handbook",
    "Gastroenterology Practice Guidelines",
    "Pulmonology Case Studies",
    "Endocrinology Reference Manual",
]

# Generic skin-related but non-specific content — for too_generic and partial_overlap
GENERIC_SKIN_FACTS = [
    "Skin is the largest organ of the human body and acts as a protective barrier.",
    "Inflammation is a common feature of many skin conditions.",
    "Skin conditions can affect people of all ages and backgrounds.",
    "The skin plays a role in temperature regulation and immune defense.",
    "Many skin disorders have both genetic and environmental components.",
    "Chronic skin conditions may have periods of remission and flare-up.",
    "Skin health is influenced by factors such as diet, stress, and hygiene.",
]

# Medication names to use in the medication_not_in_ctx edge case
MEDICATIONS_NOT_IN_CONTEXT = [
    "dupilumab", "methotrexate", "cyclosporine", "apremilast",
    "secukinumab", "ustekinumab", "doxycycline", "isotretinoin",
    "clobetasol", "tacrolimus",
]

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_primary_chunk(topic_data: dict, source: str) -> dict:
    facts = topic_data["facts"][:3]
    return {
        "priority": "PRIORITY 1",
        "source": source,
        "content": (
            f"Source: {source}\n"
            f"Topic: {topic_data['topic'].capitalize()}\n"
            + "\n".join(f"- {f}" for f in facts)
        ),
    }


def build_secondary_chunk(topic_data: dict, source: str) -> dict:
    facts = topic_data["facts"][2:] if len(topic_data["facts"]) > 2 else topic_data["facts"]
    return {
        "priority": "PRIORITY 2",
        "source": source,
        "content": (
            f"Source: {source}\n"
            + "\n".join(f"- {f}" for f in facts)
        ),
    }


def build_irrelevant_chunk(source: str) -> dict:
    irrelevant_texts = [
        "The heart pumps blood through the body. Normal resting heart rate is 60-100 bpm. High blood pressure often has no symptoms.",
        "Kidney function is measured by GFR. Chronic kidney disease progresses through 5 stages. Dialysis is used in end-stage renal disease.",
        "Bone density decreases with age. Osteoporosis increases fracture risk. Calcium and vitamin D are important for bone health.",
        "The lungs exchange oxygen and carbon dioxide. COPD is a chronic obstructive disease. Spirometry measures lung function.",
        "Blood glucose regulation involves insulin. Type 2 diabetes is linked to insulin resistance. HbA1c reflects 3-month glucose control.",
    ]
    return {
        "priority": "PRIORITY 1",
        "source": source,
        "content": f"Source: {source}\n{random.choice(irrelevant_texts)}",
    }


def build_generic_chunk(source: str) -> dict:
    """Superficially skin-related but too vague to answer any specific question."""
    facts = random.sample(GENERIC_SKIN_FACTS, k=3)
    return {
        "priority": "PRIORITY 1",
        "source": source,
        "content": (
            f"Source: {source}\nTopic: General Skin Health\n"
            + "\n".join(f"- {f}" for f in facts)
        ),
    }


def build_partial_overlap_chunk(related_topic: str, source: str) -> dict:
    """Context covers a related condition (e.g., inflammation) but not the specific one asked."""
    return {
        "priority": "PRIORITY 1",
        "source": source,
        "content": (
            f"Source: {source}\nTopic: Skin Inflammation (General)\n"
            f"- Inflammation is a key feature of many skin conditions including {related_topic}.\n"
            "- Inflammatory skin responses involve immune cell activation and cytokine release.\n"
            "- The specific mechanisms vary between different inflammatory dermatoses.\n"
            "- General management of skin inflammation includes avoiding triggers and moisturizing."
        ),
    }


def format_context_block(chunks: list, question: str) -> str:
    block = "[CONTEXT]\n"
    for chunk in chunks:
        block += f"\n## {chunk['priority']}: {chunk['source']}\n{chunk['content']}\n"
    block += "[/CONTEXT]"
    return f"{block}\n\n[QUESTION]\n{question}"


def format_empty_context_block(question: str) -> str:
    """Format used when retriever returns nothing."""
    return "[CONTEXT]\n[/CONTEXT]\n\n[QUESTION]\n" + question

# ─────────────────────────────────────────────────────────────────────────────
# GROQ CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_groq(prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def ensure_double_newline_after_think(text: str) -> str:
    return re.sub(r"</think>\s*\n+", "</think>\n\n", text)


def validated_groq(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            raw = call_groq(prompt)
            if "<think>" not in raw or "</think>" not in raw:
                raise ValueError("Missing <think> tags")
            return ensure_double_newline_after_think(raw)
        except Exception as e:
            logger.warning(f"  attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                raise


def make_record(user_content: str, assistant_content: str, metadata: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_UNIFIED},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": metadata,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION — normal cases
# ─────────────────────────────────────────────────────────────────────────────
def generate_one_chunk(topic_data: dict) -> dict:
    topic    = topic_data["topic"]
    q_type   = random.choice(list(QUESTION_TYPES.keys()))
    question = random.choice(QUESTION_TYPES[q_type]).format(topic=topic)
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)
    source   = random.choice(SOURCES_PRIMARY)
    chunk    = build_primary_chunk(topic_data, source)
    cite     = random.choice(CITATION_PHRASES).format(source=source)

    prompt = (
        "You are generating training data for a medical RAG assistant.\n\n"
        "## Context (single source):\n"
        f"{chunk['content']}\n\n"
        f"## User question:\n{question}\n\n"
        "## Requirements for the assistant response:\n"
        "1. Start with <think> tags. Inside, briefly analyze what the context says and plan your answer.\n"
        "2. After </think>, leave a blank line, then write your final answer.\n"
        f"3. Begin the answer with this citation phrase: \"{cite}, ...\"\n"
        "4. NEVER prescribe medications or give specific treatments.\n"
        "5. Keep the final answer SHORT (1-3 sentences).\n"
        f"6. End with this follow-up: \"{followup}\"\n\n"
        "Output format:\n"
        "<think>\n[reasoning]\n</think>\n\n"
        "[answer starting with citation phrase, ending with follow-up]\n\n"
        "Return ONLY the assistant response. Nothing else."
    )

    raw = validated_groq(prompt)
    return make_record(
        format_context_block([chunk], question),
        raw,
        {"type": "one_chunk", "topic": topic, "source": source},
    )


def generate_two_chunk(topic_data: dict) -> dict:
    topic    = topic_data["topic"]
    q_type   = random.choice(list(QUESTION_TYPES.keys()))
    question = random.choice(QUESTION_TYPES[q_type]).format(topic=topic)
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)
    src1     = random.choice(SOURCES_PRIMARY)
    src2     = random.choice(SOURCES_SECONDARY)
    chunk1   = build_primary_chunk(topic_data, src1)
    chunk2   = build_secondary_chunk(topic_data, src2)
    cite1    = random.choice(CITATION_PHRASES).format(source=src1)
    cite2    = random.choice([p for p in CITATION_PHRASES if p != cite1]).format(source=src2)

    prompt = (
        "You are generating training data for a medical RAG assistant.\n\n"
        "## Context sources:\n"
        f"PRIORITY 1 — {src1}:\n{chunk1['content']}\n\n"
        f"PRIORITY 2 — {src2}:\n{chunk2['content']}\n\n"
        f"## User question:\n{question}\n\n"
        "## Requirements:\n"
        "1. Start with <think>. Inside, compare both sources: priority, complementarity, what to use.\n"
        "2. After </think>, leave a blank line, then write your final answer.\n"
        f"3. Cite using: \"{cite1}\" and/or \"{cite2}\"\n"
        "4. NEVER prescribe medications or give specific treatments.\n"
        "5. Keep the final answer SHORT (1-3 sentences).\n"
        f"6. End with: \"{followup}\"\n\n"
        "Output format:\n"
        "<think>\n[reasoning comparing both sources]\n</think>\n\n"
        "[answer citing source(s), ending with follow-up]\n\n"
        "Return ONLY the assistant response. Nothing else."
    )

    raw = validated_groq(prompt)
    return make_record(
        format_context_block([chunk1, chunk2], question),
        raw,
        {"type": "two_chunk", "topic": topic,
         "source_primary": src1, "source_secondary": src2},
    )

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION — edge cases
# ─────────────────────────────────────────────────────────────────────────────

def generate_edge_irrelevant_domain(topic_data: dict) -> dict:
    """Context is from a completely different medical specialty (cardiology, nephrology, etc.)."""
    topic    = topic_data["topic"]
    question = random.choice([
        f"What should I know about {topic}?",
        f"What causes {topic}?",
        f"How is {topic} managed?",
    ])
    source  = random.choice(SOURCES_IRRELEVANT)
    chunk   = build_irrelevant_chunk(source)
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)

    assistant = (
        "<think>\n"
        f"The context comes from '{source}', which is unrelated to dermatology or {topic}. "
        "It covers a completely different medical specialty. "
        "Using this context to answer a skin-related question would be hallucination. "
        "I must clearly state that the available context does not address the user's question "
        "and avoid inventing any information.\n"
        "</think>\n\n"
        f"The information available in the provided source ({source}) is not related to {topic} "
        "or skin conditions in general. "
        "I'm unable to answer your question based on this context. "
        f"I recommend consulting a dermatologist for accurate guidance. {followup}"
    )

    return make_record(
        format_context_block([chunk], question),
        assistant,
        {"type": "edge_irrelevant_domain", "topic": topic, "source": source},
    )


def generate_edge_no_context(topic_data: dict) -> dict:
    """
    Retriever returned nothing — empty [CONTEXT] block.
    This happens when no chunk scored above the relevance threshold.
    """
    topic    = topic_data["topic"]
    question = random.choice([
        f"What is {topic}?",
        f"Can you tell me about {topic}?",
        f"What causes {topic}?",
    ])
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)

    no_context_responses = [
        (
            "<think>\n"
            "The context block is empty — the retrieval system did not find any relevant "
            f"information for this query about {topic}. "
            "I must not answer from memory or training data, as that could introduce hallucinations. "
            "I should honestly inform the user that no relevant information was found "
            "and suggest they consult a professional.\n"
            "</think>\n\n"
            f"I wasn't able to find relevant information about {topic} in my knowledge base at this time. "
            f"For accurate information, I recommend consulting a dermatologist. {followup}"
        ),
        (
            "<think>\n"
            f"No context was retrieved for the question about {topic}. "
            "The [CONTEXT] block is empty. "
            "Answering without context would mean relying on unverified training data, which risks hallucination. "
            "The correct behavior is to acknowledge the gap and redirect the user.\n"
            "</think>\n\n"
            f"I'm sorry, I couldn't retrieve any information related to your question about {topic} "
            "from my database right now. "
            f"Please consult a qualified dermatologist for guidance. {followup}"
        ),
        (
            "<think>\n"
            "Empty context block detected. "
            f"The retrieval returned no results for the query about {topic}. "
            "Without a verified source, I cannot provide a reliable answer. "
            "I should be transparent about this limitation.\n"
            "</think>\n\n"
            f"Unfortunately, my search didn't return any relevant results for '{topic}' at this moment. "
            "I can only answer based on retrieved sources, so I'd recommend checking with a dermatologist "
            f"for accurate information. {followup}"
        ),
    ]

    assistant = random.choice(no_context_responses)

    return make_record(
        format_empty_context_block(question),
        assistant,
        {"type": "edge_no_context", "topic": topic},
    )


def generate_edge_too_generic(topic_data: dict) -> dict:
    """
    Context is superficially skin-related but too generic to answer the specific question.
    Model should acknowledge the limitation rather than extrapolating.
    """
    topic    = topic_data["topic"]
    question = random.choice([
        f"What are the specific triggers of {topic}?",
        f"How does {topic} differ from other skin conditions?",
        f"What are the main symptoms of {topic}?",
    ])
    source   = random.choice(SOURCES_PRIMARY)
    chunk    = build_generic_chunk(source)
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)

    prompt = (
        "You are generating training data for a medical RAG assistant.\n\n"
        "## Context provided:\n"
        f"{chunk['content']}\n\n"
        f"## User question:\n{question}\n\n"
        "## Situation:\n"
        f"The context talks about skin health in general but does NOT mention {topic} specifically "
        "or answer the user's specific question. "
        "The context is too generic and vague to be useful.\n\n"
        "## Requirements:\n"
        "1. Start with <think>. Inside, identify that the context is too generic and does not answer the question.\n"
        "2. After </think>, leave a blank line, then write your answer.\n"
        f"3. Acknowledge that the context does not contain specific information about {topic}.\n"
        "4. Do NOT attempt to answer the question using the generic context — that would be extrapolation.\n"
        "5. Keep the response SHORT (1-2 sentences).\n"
        f"6. End with: \"{followup}\"\n\n"
        "Output format:\n"
        "<think>\n[reasoning about why context is insufficient]\n</think>\n\n"
        "[honest acknowledgment that context is too generic, plus follow-up]\n\n"
        "Return ONLY the assistant response. Nothing else."
    )

    raw = validated_groq(prompt)
    return make_record(
        format_context_block([chunk], question),
        raw,
        {"type": "edge_too_generic", "topic": topic, "source": source},
    )


def generate_edge_partial_overlap(topic_data: dict) -> dict:
    """
    Context mentions a related concept (e.g., skin inflammation) but not the
    specific condition or aspect the user is asking about.
    """
    topic    = topic_data["topic"]
    question = random.choice([
        f"What specifically causes the scaling in {topic}?",
        f"Why does {topic} affect certain body areas more than others?",
        f"What is the immune mechanism behind {topic}?",
    ])
    source   = random.choice(SOURCES_PRIMARY)
    chunk    = build_partial_overlap_chunk(topic, source)
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)

    prompt = (
        "You are generating training data for a medical RAG assistant.\n\n"
        "## Context provided:\n"
        f"{chunk['content']}\n\n"
        f"## User question:\n{question}\n\n"
        "## Situation:\n"
        f"The context mentions skin inflammation in general and references {topic} briefly, "
        f"but does NOT contain specific information to answer the user's detailed question about {topic}. "
        "There is a partial overlap — the topic is mentioned but the answer is not there.\n\n"
        "## Requirements:\n"
        "1. Start with <think>. Inside, identify what the context does and does not cover.\n"
        "2. After </think>, leave a blank line, then write your answer.\n"
        "3. Use what little the context offers, but clearly state that the specific information is not available.\n"
        "4. Do NOT fill the gap with knowledge from outside the context.\n"
        "5. Keep the response SHORT (2-3 sentences).\n"
        f"6. End with: \"{followup}\"\n\n"
        "Output format:\n"
        "<think>\n[reasoning about partial overlap]\n</think>\n\n"
        "[partial answer + honest acknowledgment of limitation + follow-up]\n\n"
        "Return ONLY the assistant response. Nothing else."
    )

    raw = validated_groq(prompt)
    return make_record(
        format_context_block([chunk], question),
        raw,
        {"type": "edge_partial_overlap", "topic": topic, "source": source},
    )


def generate_edge_multi_condition_gap(topic_data: dict) -> dict:
    """
    User asks about two conditions but context only covers one of them.
    Model should answer what it can and clearly flag the gap.
    """
    topic_a   = topic_data["topic"]
    # Pick a different topic for topic_b
    other     = [t for t in COMPLIANT_TOPICS if t["topic"] != topic_a]
    topic_b_data = random.choice(other)
    topic_b   = topic_b_data["topic"]

    question  = random.choice([
        f"Can you explain the difference between {topic_a} and {topic_b}?",
        f"How do {topic_a} and {topic_b} compare in terms of symptoms?",
        f"I have both {topic_a} and {topic_b} — what triggers them?",
    ])
    source    = random.choice(SOURCES_PRIMARY)
    chunk     = build_primary_chunk(topic_data, source)   # only covers topic_a
    followup  = random.choice(GENERAL_FOLLOWUPS).format(topic=topic_a)

    prompt = (
        "You are generating training data for a medical RAG assistant.\n\n"
        "## Context provided:\n"
        f"{chunk['content']}\n\n"
        f"## User question:\n{question}\n\n"
        "## Situation:\n"
        f"The context only contains information about {topic_a}. "
        f"It does NOT contain any information about {topic_b}. "
        f"The user is asking about both conditions.\n\n"
        "## Requirements:\n"
        "1. Start with <think>. Inside, note what is covered and what is missing.\n"
        "2. After </think>, leave a blank line, then write your answer.\n"
        f"3. Answer the part about {topic_a} using the context and cite the source.\n"
        f"4. Clearly state that you don't have context-based information about {topic_b}.\n"
        "5. Do NOT invent information about the missing condition.\n"
        "5. Keep the response SHORT (2-3 sentences).\n"
        f"6. End with: \"{followup}\"\n\n"
        "Output format:\n"
        "<think>\n[reasoning about the coverage gap]\n</think>\n\n"
        "[partial answer for covered topic + honest gap acknowledgment + follow-up]\n\n"
        "Return ONLY the assistant response. Nothing else."
    )

    raw = validated_groq(prompt)
    return make_record(
        format_context_block([chunk], question),
        raw,
        {"type": "edge_multi_condition_gap",
         "topic_covered": topic_a, "topic_missing": topic_b, "source": source},
    )


def generate_edge_out_of_scope(topic_data: dict) -> dict:
    """
    User asks something completely outside dermatology.
    Context may or may not be present but the question is irrelevant to the assistant's scope.
    """
    off_topic_questions = [
        "Can you help me with my diet plan for weight loss?",
        "What is the best treatment for high blood pressure?",
        "I have a persistent cough — what could it be?",
        "Is my knee pain related to arthritis?",
        "Can you recommend a good antidepressant?",
        "What causes migraines?",
        "How do I manage my blood sugar levels?",
    ]
    question = random.choice(off_topic_questions)
    source   = random.choice(SOURCES_PRIMARY)
    chunk    = build_primary_chunk(topic_data, source)   # skin context, unrelated to question
    followup = "Is there anything about your skin, hair, or nails I can help with?"

    assistant = (
        "<think>\n"
        "The user is asking about a condition or topic that is outside the scope of dermatology. "
        "The provided context is about skin health, which does not answer this question. "
        "My role is limited to skin, hair, and nail concerns. "
        "I should politely clarify my scope and offer to help with dermatology questions instead. "
        "I must not attempt to answer medical questions outside my specialty.\n"
        "</think>\n\n"
        "I'm a dermatology assistant and can only help with questions about skin, hair, and nail conditions. "
        "For your question, I'd recommend consulting the appropriate medical specialist. "
        f"{followup}"
    )

    return make_record(
        format_context_block([chunk], question),
        assistant,
        {"type": "edge_out_of_scope", "question_preview": question[:60]},
    )


def generate_edge_medication_not_in_ctx(topic_data: dict) -> dict:
    """
    User asks about a specific medication by name that does not appear in context.
    Model must not invent information about the drug.
    """
    topic    = topic_data["topic"]
    drug     = random.choice(MEDICATIONS_NOT_IN_CONTEXT)
    question = random.choice([
        f"Is {drug} effective for {topic}?",
        f"What are the side effects of {drug}?",
        f"My doctor mentioned {drug} for {topic} — what can you tell me about it?",
        f"How does {drug} work for {topic}?",
    ])
    source   = random.choice(SOURCES_PRIMARY)
    chunk    = build_primary_chunk(topic_data, source)   # context about topic, no drug info
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)

    prompt = (
        "You are generating training data for a medical RAG assistant.\n\n"
        "## Context provided:\n"
        f"{chunk['content']}\n\n"
        f"## User question:\n{question}\n\n"
        "## Situation:\n"
        f"The context contains general information about {topic} but does NOT mention "
        f"the medication '{drug}' at all. "
        "The user is asking specifically about this drug.\n\n"
        "## Requirements:\n"
        "1. Start with <think>. Inside, confirm the drug is absent from context.\n"
        "2. After </think>, leave a blank line, then write your answer.\n"
        f"3. Do NOT provide any information about {drug} — not even general pharmacology.\n"
        "4. Clearly state that the context does not contain information about this medication.\n"
        "5. Recommend consulting a dermatologist or pharmacist.\n"
        "5. Keep the response SHORT (1-2 sentences).\n"
        f"6. End with: \"{followup}\"\n\n"
        "Output format:\n"
        "<think>\n[confirm drug absent from context]\n</think>\n\n"
        "[honest response + recommendation + follow-up]\n\n"
        "Return ONLY the assistant response. Nothing else."
    )

    raw = validated_groq(prompt)
    return make_record(
        format_context_block([chunk], question),
        raw,
        {"type": "edge_medication_not_in_ctx",
         "topic": topic, "drug": drug, "source": source},
    )


def generate_edge_second_opinion(topic_data: dict) -> dict:
    """
    User wants the model to override or contradict their dermatologist's advice.
    Model must support the doctor's authority and not override medical guidance.
    """
    topic    = topic_data["topic"]
    question = random.choice([
        f"My dermatologist said I have {topic} but I don't think that's right — what do you think?",
        f"My doctor prescribed something for {topic} but I read online it's not good. Should I stop?",
        f"I disagree with my dermatologist's diagnosis of {topic}. Can you give me a second opinion?",
        f"My doctor said my {topic} needs treatment but I think it will go away on its own. Am I right?",
    ])
    source   = random.choice(SOURCES_PRIMARY)
    chunk    = build_primary_chunk(topic_data, source)
    followup = random.choice(GENERAL_FOLLOWUPS).format(topic=topic)

    assistant = (
        "<think>\n"
        "The user is asking me to contradict or override their dermatologist's professional advice. "
        "I should never do this — a dermatologist has examined the patient directly, which I cannot do. "
        "Even if the context contains relevant information, I cannot and should not replace professional "
        "medical judgment. "
        "I can share general context-based information but must clearly reinforce that the dermatologist's "
        "guidance takes priority.\n"
        "</think>\n\n"
        "I'm not able to provide a second opinion or override your dermatologist's assessment — "
        "they have examined you directly, which gives them information I don't have access to. "
        "If you have concerns about your diagnosis or treatment plan, I'd strongly encourage "
        f"discussing them directly with your dermatologist. {followup}"
    )

    return make_record(
        format_context_block([chunk], question),
        assistant,
        {"type": "edge_second_opinion", "topic": topic},
    )

# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
EDGE_GENERATORS = {
    "irrelevant_domain":      generate_edge_irrelevant_domain,
    "no_context":             generate_edge_no_context,
    "too_generic":            generate_edge_too_generic,
    "partial_overlap":        generate_edge_partial_overlap,
    "multi_condition_gap":    generate_edge_multi_condition_gap,
    "out_of_scope":           generate_edge_out_of_scope,
    "medication_not_in_ctx":  generate_edge_medication_not_in_ctx,
    "second_opinion":         generate_edge_second_opinion,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(records: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records → {output_path}")

# Split dataset into train/val/test with stratification by 'type' to ensure balanced representation of all categories in each split.
def stratified_split_by_type(records: list, train_ratio: float = 0.8, 
                              val_ratio: float = 0.1, test_ratio: float = 0.1,
                              seed: int = 42) -> tuple:
    """Split records stratified by metadata 'type' for balanced distribution."""
    random.seed(seed)
    
    by_type = defaultdict(list)
    for r in records:
        by_type[r["metadata"]["type"]].append(r)
    
    train, val, test = [], [], []
    
    for dtype, items in by_type.items():
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
        
        logger.info(f"  {dtype}: {n_train} train, {n_val} val, {n_test} test")
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate Dataset C — QA-RAG with reasoning (Groq)")
    ap.add_argument("--num_examples", type=int, default=100)
    ap.add_argument("--output",       type=str, default=None)
    args = ap.parse_args()

    output_path = args.output or os.path.join(ROOT, "prepare_datasets", "temporal_datasets", "dataset_c")
   

    # Distribution:
    #   35% one-chunk    (standard single-source RAG)
    #   35% two-chunk    (multi-source comparison)
    #   30% edge cases   (spread evenly across 8 types)
    n_one  = int(args.num_examples * 0.35)
    n_two  = int(args.num_examples * 0.35)
    n_edge = args.num_examples - n_one - n_two

    # Distribute edge budget as evenly as possible across all 8 types
    edge_types  = list(EDGE_GENERATORS.keys())
    n_types     = len(edge_types)
    base_per    = n_edge // n_types
    remainder   = n_edge % n_types
    edge_counts = {et: base_per + (1 if i < remainder else 0)
                   for i, et in enumerate(edge_types)}

    logger.info(f"Provider  : Groq / {MODEL}")
    logger.info(f"Plan      : {n_one} one-chunk + {n_two} two-chunk + {n_edge} edge ({n_types} types)")
    for et, count in edge_counts.items():
        logger.info(f"           edge/{et}: {count}")

    records, errors = [], 0

    # ── Normal: 1-chunk ──────────────────────────────────────────────────────
    for i in range(n_one):
        td = random.choice(COMPLIANT_TOPICS)
        logger.info(f"[1-chunk {i+1:03d}/{n_one}] {td['topic']}")
        try:
            records.append(generate_one_chunk(td))
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"FAILED 1-chunk {i+1}: {e}")
            errors += 1

    # ── Normal: 2-chunk ──────────────────────────────────────────────────────
    for i in range(n_two):
        td = random.choice(COMPLIANT_TOPICS)
        logger.info(f"[2-chunk {i+1:03d}/{n_two}] {td['topic']}")
        try:
            records.append(generate_two_chunk(td))
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"FAILED 2-chunk {i+1}: {e}")
            errors += 1

    # ── Edge cases ────────────────────────────────────────────────────────────
    for et, count in edge_counts.items():
        generator = EDGE_GENERATORS[et]
        for i in range(count):
            td = random.choice(COMPLIANT_TOPICS)
            logger.info(f"[edge/{et} {i+1:02d}/{count}]")
            try:
                records.append(generator(td))
                time.sleep(0.4)
            except Exception as e:
                logger.error(f"FAILED edge/{et} {i+1}: {e}")
                errors += 1

    if not records:
        logger.error("No records generated — check GROQ_API_KEY")
        return

        # Split into train/val/test
    train_records, val_records, test_records = stratified_split_by_type(
        records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    os.makedirs(output_path, exist_ok=True)

    train_path = os.path.join(output_path, "temp_train.jsonl")
    val_path = os.path.join(output_path, "temp_val.jsonl")
    test_path = os.path.join(output_path, "temp_test.jsonl")


    save_dataset(train_records, train_path)
    save_dataset(val_records, val_path)
    save_dataset(test_records, test_path)
    
    logger.info(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    type_counts: dict = {}
    for r in records:
        k = r["metadata"]["type"]
        type_counts[k] = type_counts.get(k, 0) + 1

    print("\n" + "=" * 65)
    print("GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Total      : {len(records)}")
    for k, v in sorted(type_counts.items()):
        print(f"  {k:<35}: {v}")
    print(f"  Errors     : {errors}")
    print(f"  Output     : {output_path}")
    print("\n── Sample (first record, assistant turn) ──")
    content = records[0]["messages"][-1]["content"]
    print(content[:400] + ("..." if len(content) > 400 else ""))


if __name__ == "__main__":
    main()