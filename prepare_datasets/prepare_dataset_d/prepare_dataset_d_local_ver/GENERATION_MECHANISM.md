# Dataset D Generation Mechanism: Local Transformer Approach

This document explains the technical architecture and logic behind `generate_local_d.py`, which transforms raw clinical data into a multimodal training dataset.

## 1. Core Data Integration
The script acts as an orchestrator, pulling together three distinct "ingredients" for every training record:
- **Ground Truth Clinical Notes**: Extracted from `medical_notes_dataset`, containing detailed morphology, symptoms, and confirmed diagnoses.
- **Vision Model (DenseNet-121) Predictions**: Pre-computed classification outputs (Top-1 and Top-2 classes with confidence scores) for the associated skin image.
- **Image Metadata**: Path and labels used to synchronize the visual and textual data.

## 2. The "Combined Request" Strategy
To maximize efficiency and coherence, the script uses a single-pass generation strategy. Instead of separate calls for dialogue and reasoning, it asks the LLM for a single JSON object containing:

### Task 1: Symptom Dialogue (`dialogue` key)
- **Natural Language Simulation**: The LLM simulates 3-5 turns of a patient-assistant conversation.
- **Patient Persona**: Uses non-clinical language (e.g., "bumpy and itchy" instead of "pruritic papules").
- **Assistant Persona**: Follows strict clinical protocols (asking one question at a time about location, duration, and triggers).
- **Termination**: Ends with a specific token: `[END_SYMPTOMS]`.

### Task 2: Joint Reasoning (`assistant_content` key)
- **Internal Chain-of-Thought (`<think>`)**: The model performs a step-by-step analysis, explicitly comparing the DenseNet visual findings with the symptoms reported in the simulated dialogue.
- **Final Output (`[DIAGNOSIS]`)**: An empathetic, structured response providing a preliminary assessment, advice, and a medical disclaimer.

## 3. Local Inference Optimizations
Since the script runs on a local GPU (using a 4B model), it employs several advanced techniques to ensure reliability and speed:

- **4-Bit Quantization (BitsAndBytes)**: Compresses the model weight from ~8GB down to ~3.5GB of VRAM, allowing it to run on consumer-grade hardware.
- **Few-Shot Learning**: Injects a complete "perfect example" into the prompt. This "shows" the model the exact JSON structure and tone expected.
- **Response Pre-filling**: The script manually appends the opening brace `{` to the assistant's response. This "locks" the model into JSON mode immediately, preventing conversational "chatter" (e.g., "Here is your JSON...").
- **Low Temperature (0.2)**: Reduces randomness to ensure the model prioritizes structural validity over creative variation.

## 4. Robust JSON Extraction
Local models often append extra text or explanations after the closing brace. The script handles this using a "surgical" parsing method:
- **`json.JSONDecoder().raw_decode()`**: This Python utility parses the first valid JSON object it finds and **stops exactly at the closing `}`**. It ignores any trailing text that would normally cause a `JSONDecodeError`.

## 5. Training Record Construction
Once the JSON is parsed, the script reconstructs the full message history required for training:
1. **System Prompt**: Injects the `SYSTEM_UNIFIED` clinical guidelines.
2. **Dialogue**: Appends the generated turns.
3. **Image Indexing**: Calculates the `image_msg_index`. This marks the precise point (after symptom collection) where visual tokens must be injected during multimodal training.
4. **Final Response**: Appends the reasoning and diagnosis blocks.

## Summary
By combining vision model outputs with simulated clinical conversations, this mechanism teaches the final model not just to "see" images, but to **reason** through a patient's history and visual evidence simultaneously.
