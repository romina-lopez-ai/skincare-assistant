"""Dataset B — Dermatology symptom-collection dialogue dataset.
Provider : GROQ — Llama 3.3 70B
Usage:
    pip install groq python-dotenv
    python dataset_b_groq.py --num_examples 400 --edge_ratio 0.3
"""

import json, os, re, time, logging, random, argparse,sys
from groq import Groq
from dotenv import load_dotenv
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
# SEED DATA
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS = [
    "acne","psoriasis","eczema","seborrheic dermatitis","rosacea",
    "contact dermatitis","urticaria","tinea corporis","vitiligo",
    "actinic keratosis","cellulitis","perioral dermatitis","lichen planus",
]
LOCATIONS = [
    "face","jawline","chin","cheeks","forehead","scalp","neck","chest",
    "upper back","lower back","abdomen","elbows","knees","hands","feet",
    "palms","soles","underarms","groin","eyelids","lips",
]
SYMPTOM_TYPES = [
    "cystic acne","red papules","pustules","vesicles","bullae","dry patches",
    "scaly plaques","silvery scales","oozing lesions","crusts","ulcers",
    "nodules","hives","blanching rash","non-blanching rash",
    "hyperpigmentation","hypopigmentation",
]

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING BLOCKS
# These are plain strings — .format() is never called on them because they
# contain JSON examples with { } that would break str.format().
# Prompts are assembled via string concatenation only.
# ─────────────────────────────────────────────────────────────────────────────

IDENTITY_BLOCK = """\
ASSISTANT IDENTITY (applies to every message the assistant writes):
- You are a dermatology symptom-collection assistant.
- Your ONLY job is to ask questions and collect symptom information.
- You NEVER diagnose, recommend treatments, prescribe medications,
  suggest tests, refer to other doctors, or give any medical advice.
- You NEVER mention phone numbers, clinics, appointments, or suggest
  the patient seek care anywhere else.
- You NEVER switch language — English only.
- Every assistant message is either a single question to collect more
  information, or the final closing [END_SYMPTOMS] message.
- If the patient goes off-topic, gently redirect them to describing
  their skin symptoms and ask the next question.\
"""

FORMAT_BLOCK = """\
OUTPUT FORMAT — MANDATORY
=========================
Return ONLY a raw JSON array. No markdown. No code fences. No explanation.
Start immediately with [ and end with ]. Nothing outside the brackets.

Each element is a JSON object with exactly two string fields:
  "role"    -> must be "user" or "assistant" (no other value allowed)
  "content" -> the spoken text

Rules:
- Roles MUST strictly alternate: user, assistant, user, assistant, ...
- NEVER two consecutive messages with the same role.
- The LAST message must have role "assistant" and must contain [END_SYMPTOMS].
- The closing phrase can vary, e.g.:
    "Thank you for sharing all of that.[END_SYMPTOMS]"
    "I now have everything I need.[END_SYMPTOMS]"
    "That covers everything, thank you.[END_SYMPTOMS]"
  but [END_SYMPTOMS] is always required.

Minimal correct example:
[
  {"role": "user", "content": "I have a rash on my arm."},
  {"role": "assistant", "content": "When did it first appear?"},
  {"role": "user", "content": "About two weeks ago."},
  {"role": "assistant", "content": "Thank you for sharing all of that.[END_SYMPTOMS]"}
]\
"""

CLINICAL_DOMAINS = """\
The assistant must cover ALL 12 clinical domains before ending:
 1. Onset & duration
 2. Morphology (size, shape, color, texture, borders)
 3. Distribution & spread (localized vs spreading, symmetry)
 4. Sensations (itch 1-10, pain, burning, tingling)
 5. Temporal patterns (constant vs intermittent, time of day)
 6. Aggravating factors (heat, sweat, sun, stress, friction, food)
 7. Alleviating factors (what helps, even partially)
 8. Prior treatments and their effect (OTC, home remedies)
 9. Triggers & recent exposures (new products, travel, diet changes)
10. Systemic symptoms (fever, fatigue, joint pain, lymph nodes)
11. Personal history (allergies, atopy, prior skin conditions)
12. Family history (eczema, psoriasis, skin cancer)\
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_normal_prompt(condition: str, location: str, symptom: str) -> str:
    task = (
        "TASK\n====\n"
        "Generate a realistic English-only dermatology chat dialogue for a dataset.\n"
        "The assistant only collects symptoms from a patient about a skin condition.\n\n"
        f"Condition hint : {condition}\n"
        f"Body location  : {location}\n"
        f"Initial symptom: {symptom}\n"
    )
    
    rules = (
        "\nCONVERSATION RULES:\n"
        '- Patient ("user") opens with a 1-3 sentence natural complaint.\n'
        "\n"
        "--- QUESTION STYLE VARIETY ---\n"
        "Mix these question types. Do NOT use the same pattern repeatedly:\n"
        "  • Direct:        'When did it first appear?'\n"
        "  • A/B choice:    'Is it more painful or itchy?'\n"
        "  • Rating:        'On a scale of 1-10, how intense?'\n"
        "  • Clarifying:    'Can you describe what it looks like?'\n"
        "  • Confirmation:  'So it started about two weeks ago?'\n"
        "  • Implied:       'And the redness — has it spread?'\n"
        "  • Empathetic:    'I understand. Have you noticed any triggers?'\n"
        "  • Short:         'Any other symptoms?'\n"
        "  • Progressive:   'What about other areas?'\n"
        "\n"
        "--- AVOID VERBATIM REPETITION ---\n"
        "Do NOT repeat what the patient just said. Use implied context.\n"
        "    Patient: 'two weeks ago' → Assistant: 'It started two weeks ago?'\n"
        "    Patient: 'two weeks ago' → Assistant: 'Did it come on suddenly?'\n"
        "\n"
        "--- VARY SENTENCE OPENERS ---\n"
        "Mix openers: When, Where, How, What about, Can you, Tell me about, And, So, Is there, Do you.\n"
        "\n"
        "--- MESSAGE LENGTH VARIETY ---\n"
        "Patient messages:\n"
        "  • Very short (1-5 words):    'Yes.', 'No.', 'Not really.'\n"
        "  • Short (6-15 words):        'About two weeks ago.'\n"
        "  • Medium (16-40 words):      'It gets worse at night when I'm trying to sleep.'\n"
        "  • Long (41-100 words):       'I tried hydrocortisone for a week but it didn't help.'\n"
        "\n"
        "Assistant messages:\n"
        "  • Very short:  'Any fever?', 'How long ago?'\n"
        "  • Short:       'Does it itch or burn?'\n"
        "  • Medium:      'I understand. Have you noticed any triggers?'\n"
        "  • Longer:      'So let me summarize: rash for 2 weeks, itchy, only on arms. Correct?'\n"
        "\n"
        "--- FLOW ---\n"
        "- Roles strictly alternate: user, assistant, user, assistant, ...\n"
        "- No two consecutive messages from the same role.\n"
        "- Assistant asks 8-14 questions total.\n"
        "- End with [END_SYMPTOMS] after the last assistant message. This is mandatory.\n"
    )
    
    return "\n\n".join([IDENTITY_BLOCK, task + CLINICAL_DOMAINS + rules, FORMAT_BLOCK])


def build_edge_prompt(edge_type: str) -> str:
    header = (
        IDENTITY_BLOCK + "\n\n"
        "TASK\n====\n"
        "Generate a realistic English-only dermatology chat dialogue for a dataset.\n"
    )

    bodies = {

        "non_committal": (
            "EDGE CASE: the patient gives vague, uncertain answers most of the time.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Opens with a real skin complaint (1-2 sentences).\n"
            "- For most (not all) answers uses phrases like:\n"
            '  "mm", "I\'m not sure", "maybe", "hard to say", "I don\'t know",\n'
            '  "kind of?", "I think so but I\'m not certain", "possibly".\n'
            "- Occasionally gives a clearer answer — not every reply is vague.\n\n"
            "ASSISTANT BEHAVIOR:\n"
            "- Asks ONE question per turn — never two.\n"
            "- Shows brief empathy when patient is vague.\n"
            "- Rephrases hard questions as A/B choices when needed.\n"
            "- Covers as many of the 12 clinical domains as possible.\n"
            "- Ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Mix short and long messages. Total messages: 12-18."
        ),

        "nonsense_input": (
            "EDGE CASE: the patient accidentally sends gibberish at ONE point.\n\n"
            "RULES:\n"
            "- Conversation STARTS normally with a real skin complaint.\n"
            "- At exactly ONE point in the middle the patient sends gibberish\n"
            '  (e.g. "asdfjkl;", "sdfg 123 @@", "qwerty uiop", random symbols).\n'
            "- The assistant replies it did not understand and asks the patient\n"
            "  to rephrase — one polite sentence only, no other suggestions.\n"
            "- The patient corrects themselves and the conversation continues normally.\n"
            "- ONLY 1 gibberish message in the whole conversation.\n"
            "- Assistant ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Mix short and long messages. Total messages: 12-18."
        ),

        "out_of_domain": (
            "EDGE CASE: the patient asks about something unrelated to skin.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Opens by describing a NON-skin problem:\n"
            "  chest pain, headache, knee pain, dizziness, stomach ache, anxiety, etc.\n\n"
            "ASSISTANT BEHAVIOR:\n"
            "- Politely explains it only collects information about skin, hair, and nail concerns.\n"
            "- Does NOT suggest calling anyone, going anywhere, or seeking other care.\n"
            "- Asks once if the patient also has any skin concern.\n"
            "- If patient says yes: collects skin symptoms normally and ends with [END_SYMPTOMS].\n"
            "- If patient says no: closes politely with [END_SYMPTOMS].\n\n"
            "Total messages: 6-10."
        ),

        "frustrated_user": (
            "EDGE CASE: the patient gets frustrated at 1-2 points during the conversation.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Starts with a normal skin complaint.\n"
            "- MOST messages are normal answers.\n"
            "- Only 1 or 2 messages express frustration:\n"
            '  "Why do you need to know all this?", "I already told you that.",\n'
            '  "This is taking forever.", "Can you just help me already?"\n'
            "- After acknowledgment, continues normally.\n\n"
            "ASSISTANT BEHAVIOR:\n"
            "- One brief calm acknowledgment then immediately asks the next question.\n"
            "- Never defensive, never skips domains.\n"
            "- Ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Mix short and long messages. Total messages: 14-20."
        ),

        "contradictory": (
            "EDGE CASE: the patient unintentionally contradicts themselves once.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Starts with a normal skin complaint.\n"
            "- At ONE point contradicts something said earlier:\n"
            '  - Said "no pain at all" then later "it actually hurts when I press it"\n'
            '  - Said "started two weeks ago" then later "I\'ve had this for years"\n'
            '  - Said "just on my face" then later "and a bit on my neck too"\n\n'
            "ASSISTANT BEHAVIOR:\n"
            "- Gently points out the contradiction without making the patient feel bad.\n"
            "- Waits for clarification before continuing.\n"
            "- Ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Mix short and long messages. Total messages: 14-20."
        ),

        "incomplete_answer": (
            "EDGE CASE: the patient gives very short answers throughout.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Opens with a short skin complaint (1-2 sentences).\n"
            '- Most answers are very brief: "yes", "no", "not sure", "a bit", "ok", "maybe".\n'
            "- Some answers are slightly longer (1-2 sentences).\n\n"
            "ASSISTANT BEHAVIOR:\n"
            "- Keeps its own messages short and specific.\n"
            "- Uses A/B choices when answers are too brief.\n"
            "- One clarifying follow-up before moving on if needed.\n"
            "- Ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Total messages: 14-20."
        ),

        "self_diagnosis": (
            "EDGE CASE: the patient arrives convinced of their own diagnosis.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Opens by stating their self-diagnosis and asking for treatment:\n"
            '  "I googled it, it\'s psoriasis — what cream should I use?"\n'
            '  "I\'m pretty sure it\'s eczema, I just need the right prescription."\n'
            "- May push back once or twice when redirected.\n"
            "- Eventually cooperates and answers all questions.\n\n"
            "ASSISTANT BEHAVIOR:\n"
            "- Acknowledges the research respectfully — never dismisses it.\n"
            "- Explains clearly that its role is only to collect symptom information.\n"
            "- Redirects to systematic symptom questions and covers all 12 clinical domains.\n"
            "- Ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Mix short and long messages. Total messages: 16-22."
        ),

        "multiple_conditions": (
            "EDGE CASE: the patient mentions 2-3 separate skin problems at once.\n\n"
            "PATIENT BEHAVIOR:\n"
            "- Opens by listing multiple skin concerns:\n"
            '  e.g. "I have a rash on my arm, a weird mole on my back, and dry patches on my scalp."\n\n'
            "ASSISTANT BEHAVIOR:\n"
            "- Acknowledges all concerns briefly.\n"
            "- Explains it will address one at a time, starting with the most bothersome.\n"
            "- Collects complete symptoms for the first concern (all 12 clinical domains).\n"
            "- Notes the others briefly before closing.\n"
            "- Ends with [END_SYMPTOMS].\n\n"
            + CLINICAL_DOMAINS + "\n\n"
            "Mix short and long messages. Total messages: 18-24."
        ),
    }

    return header + bodies[edge_type] + "\n\n" + FORMAT_BLOCK


EDGE_TYPES = [
    "non_committal", "nonsense_input", "out_of_domain", "frustrated_user",
    "contradictory", "incomplete_answer", "self_diagnosis", "multiple_conditions",
]

# ─────────────────────────────────────────────────────────────────────────────
# ROLE NORMALISATION MAP
# ─────────────────────────────────────────────────────────────────────────────
ROLE_MAP = {
    "doctor":    "assistant", "Doctor":    "assistant", "DOCTOR":    "assistant",
    "physician": "assistant", "Physician": "assistant",
    "assistant": "assistant", "Assistant": "assistant", "ASSISTANT": "assistant",
    "patient":   "user",      "Patient":   "user",      "PATIENT":   "user",
    "user":      "user",      "User":      "user",      "USER":      "user",
}

# ─────────────────────────────────────────────────────────────────────────────
# GROQ API CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_groq(prompt: str, temperature: float = 0.85, max_tokens: int = 4000) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_json_array(raw: str) -> list:
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text).strip()

    def try_parse(s: str):
        data = json.loads(s)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("messages", "dialogue", "conversation", "data", "chat"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            for v in data.values():
                if isinstance(v, list):
                    return v
        return None

    try:
        result = try_parse(text)
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            result = try_parse(text[start:end + 1])
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No JSON array found in response (first 300 chars): {raw[:300]}")

# ─────────────────────────────────────────────────────────────────────────────
# ROLE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def normalise_roles(dialogue: list) -> list:
    out = []
    for i, msg in enumerate(dialogue):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} is not a dict: {msg!r}")
        raw_role = msg.get("role") or msg.get("speaker") or msg.get("type") or ""
        if not raw_role:
            raw_role = "user" if i % 2 == 0 else "assistant"
        normed = ROLE_MAP.get(str(raw_role))
        if normed is None:
            raise ValueError(f"Unrecognised role '{raw_role}' at message {i}")
        content = msg.get("content") or msg.get("text") or msg.get("message") or ""
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": normed, "content": content.strip()})
    return out

# ─────────────────────────────────────────────────────────────────────────────
# FIX CONSECUTIVE ROLES
# ─────────────────────────────────────────────────────────────────────────────
def fix_consecutive_roles(dialogue: list) -> list:
    if not dialogue:
        return dialogue
    merged = [dialogue[0].copy()]
    for msg in dialogue[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] = merged[-1]["content"].rstrip() + " " + msg["content"]
        else:
            merged.append(msg.copy())
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate(dialogue: list, require_end: bool = True,
             min_patient_turns: int = 3, check_coverage: bool = False) -> bool:
    if not dialogue:
        logger.warning("Validation: empty dialogue"); return False
    if dialogue[0]["role"] != "user":
        logger.warning("Validation: must start with user"); return False
    for i, m in enumerate(dialogue):
        if not m["content"].strip():
            logger.warning(f"Validation: empty content at message {i}"); return False
    for i in range(len(dialogue) - 1):
        if dialogue[i]["role"] == dialogue[i + 1]["role"]:
            logger.warning(f"Validation: consecutive same role at {i}"); return False
    for i, m in enumerate(dialogue):
        if m["role"] == "assistant" and m["content"].count("?") > 2:
            logger.warning(f"Validation: >2 questions in assistant turn {i}"); return False
    if require_end:
        last = dialogue[-1]
        if last["role"] != "assistant" or "[END_SYMPTOMS]" not in last["content"]:
            logger.warning("Validation: missing [END_SYMPTOMS] in last message"); return False
    pt = sum(1 for m in dialogue if m["role"] == "user")
    if pt < min_patient_turns:
        logger.warning(f"Validation: {pt} patient turns < minimum {min_patient_turns}"); return False
    if check_coverage:
        full    = " ".join(m["content"] for m in dialogue).lower()
        covered = sum(1 for kw in ["when","how long","itch","pain","treat","family","worse","better"] if kw in full)
        if covered < 4:
            logger.warning(f"Validation: clinical coverage {covered}/8"); return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def generate_normal(condition: str, location: str, symptom: str, retries: int = 3) -> list:
    prompt = build_normal_prompt(condition, location, symptom)
    for attempt in range(retries):
        try:
            raw = call_groq(prompt, temperature=0.85)
            dlg = fix_consecutive_roles(normalise_roles(extract_json_array(raw)))
            if len(dlg) < 10:
                raise ValueError(f"Too short: {len(dlg)} turns")
            if not validate(dlg, require_end=True, min_patient_turns=5, check_coverage=True):
                raise ValueError("Quality check failed")
            return dlg
        except Exception as e:
            logger.warning(f"Normal attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                raise


def generate_edge(edge_type: str, retries: int = 4) -> list:
    prompt    = build_edge_prompt(edge_type)
    min_turns = 2 if edge_type == "out_of_domain" else 3
    for attempt in range(retries):
        try:
            raw = call_groq(prompt, temperature=0.9)
            dlg = fix_consecutive_roles(normalise_roles(extract_json_array(raw)))
            if len(dlg) < 4:
                raise ValueError(f"Too short: {len(dlg)} turns")
            if not validate(dlg, require_end=True, min_patient_turns=min_turns, check_coverage=False):
                raise ValueError("Quality check failed")
            return dlg
        except Exception as e:
            logger.warning(f"Edge '{edge_type}' attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                raise

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(records: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records → {output_path}")

# ============================================================
# STRATIFIED SPLIT FUNCTION
# ============================================================

def stratified_split_by_type(records: list, train_ratio: float = 0.8, 
                              val_ratio: float = 0.1, test_ratio: float = 0.1,
                              seed: int = 42) -> tuple:
    """
    Split records stratified by metadata 'type' (normal vs edge) and 'edge_type'.
    Ensures balanced distribution across train/val/test.
    Returns (train_records, val_records, test_records)
    """
    random.seed(seed)
    
    # Group records by type for stratification
    # Normal records go together, each edge type goes separately
    from collections import defaultdict
    by_type = defaultdict(list)
    
    for r in records:
        if r["metadata"]["type"] == "normal":
            by_type["normal"].append(r)
        else:
            # Group by specific edge_type for better balance
            edge_type = r["metadata"]["edge_type"]
            by_type[f"edge_{edge_type}"].append(r)
    
    train, val, test = [], [], []
    
    for group_name, items in by_type.items():
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
        
        logger.info(f"  {group_name}: {n_train} train, {n_val} val, {n_test} test")
    
    # Shuffle final splits to mix types
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate dermatology symptom-collection dialogues (Groq)")
    ap.add_argument("--num_examples", type=int,   default=100)
    ap.add_argument("--edge_ratio",   type=float, default=0.3)
    ap.add_argument("--output",       type=str,   default=None)
    ap.add_argument("--output_dir",   type=str,   default=None,
                    help="Directory to save train/val/test splits (overrides --output)")
    args = ap.parse_args()

    # Determine output paths
    if args.output_dir:
        output_dir = args.output_dir
    elif args.output:
        output_dir = os.path.dirname(args.output)
    else:
        output_dir = os.path.join(ROOT, "prepare_datasets", "temporal_datasets", "dataset_b")
  
    num_edge    = int(args.num_examples * args.edge_ratio)
    num_normal  = args.num_examples - num_edge

    logger.info(f"Provider : Groq / {MODEL}")
    logger.info(f"Plan     : {num_normal} normal + {num_edge} edge = {args.num_examples}")
    logger.info(f"Output dir: {output_dir}")

    records, errors = [], 0

    # Normal dialogues
    for i in range(num_normal):
        cond = random.choice(CONDITIONS)
        loc  = random.choice(LOCATIONS)
        sym  = random.choice(SYMPTOM_TYPES)
        logger.info(f"[Normal {i + 1:03d}/{num_normal}] {cond} | {loc} | {sym}")
        try:
            dlg = generate_normal(cond, loc, sym)
            records.append({
                "messages": [{"role": "system", "content": SYSTEM_UNIFIED}] + dlg,
                "metadata": {
                    "type": "normal", "edge_type": None,
                    "condition": cond, "location": loc, "symptom_seed": sym,
                    "provider": "groq", "model": MODEL,
                },
            })
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"FAILED normal {i + 1}: {e}")
            errors += 1

    # Edge dialogues
    for i in range(num_edge):
        et = random.choice(EDGE_TYPES)
        logger.info(f"[Edge  {i + 1:03d}/{num_edge}] type={et}")
        try:
            dlg = generate_edge(et)
            records.append({
                "messages": [{"role": "system", "content": SYSTEM_UNIFIED}] + dlg,
                "metadata": {
                    "type": "edge", "edge_type": et,
                    "condition": None, "location": None, "symptom_seed": None,
                    "provider": "groq", "model": MODEL,
                },
            })
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"FAILED edge {i + 1} ({et}): {e}")
            errors += 1

    if not records:
        logger.error("No records generated — check GROQ_API_KEY")
        return

    # ============================================================
    # SPLIT INTO TRAIN/VAL/TEST (80/10/10)
    # ============================================================
    train_records, val_records, test_records = stratified_split_by_type(
        records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    # Save splits
    train_path = os.path.join(output_dir, "temp_train.jsonl")
    val_path = os.path.join(output_dir, "temp_val.jsonl")
    test_path = os.path.join(output_dir, "temp_test.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    save_dataset(train_records, train_path)
    save_dataset(val_records, val_path)
    save_dataset(test_records, test_path)
    
    # Print statistics
    normal_n = sum(1 for r in records if r["metadata"]["type"] == "normal")
    edge_cnt: dict = {}
    for r in records:
        if r["metadata"]["type"] == "edge":
            k = r["metadata"]["edge_type"]
            edge_cnt[k] = edge_cnt.get(k, 0) + 1
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE — WITH TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    print(f"  Total generated  : {len(records)}")
    print(f"  Normal           : {normal_n}")
    print(f"  Edge             : {len(records) - normal_n}")
    for k, v in sorted(edge_cnt.items()):
        print(f"    {k:<25}: {v}")
    print(f"  Errors           : {errors}")
    print()
    print("  ── Split results ──")
    print(f"  temp_train.jsonl      : {len(train_records)} examples")
    print(f"  temp_val.jsonl        : {len(val_records)} examples")
    print(f"  test.jsonl       : {len(test_records)} examples")
    print()
    print(f"  Output directory : {output_dir}")

if __name__ == "__main__":
    main()