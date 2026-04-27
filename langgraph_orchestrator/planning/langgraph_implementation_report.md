# LangGraph Orchestration Development Report

## Overview
This implementation focused on designing and improving the LangGraph-based orchestration layer for the multimodal dermatology chatbot. The aim was to make the workflow reliable, testable, and ready for later integration with the real fine-tuned model, vision pipeline, and RAG component.

## Initial Understanding
First, analysed the project architecture and clarified how the system is structured:
- fine-tuned LLM as the main conversational engine
- DenseNet as the image analysis component
- projection layer for image-to-LLM embedding alignment
- RAG as a separate follow-up knowledge component
- LangGraph as the orchestration layer controlling workflow and state

The model could ask too many symptom questions, produce malformed output tags, and behave inconsistently across phases. This confirmed that a strong orchestration layer was necessary.

## Workflow Planning
Created the initial orchestration design including:
- system components
- conversation states
- node structure
- routing rules
- session memory design
- example user flows

The main states were defined as:
- `start`
- `symptom_collection`
- `await_image`
- `joint_diagnosis`
- `follow_up`
- `end_session`

The main nodes included:
- session entry
- input router
- symptom collection
- image handling
- vision processing
- projection
- diagnosis
- response cleaning
- follow-up router
- context retrieval
- RAG answer
- session end

## Implementation Progress
Created the LangGraph workflow skeleton locally using a new virtual environment and a dedicated branch. The initial implementation included:
- state schema
- placeholder nodes
- routing logic
- graph builder
- web testing interface

This first version allowed to test the flow visually, even before connecting the real model.

## Phase 1 — Service Layer Abstraction
The first improvement was to separate orchestration logic from backend/model logic.

### What was done
I refactored the workflow so that nodes only handled:
- state updates
- routing
- phase transitions

Mock backend logic was moved into service files:
- `llm_service.py`
- `vision_service.py`
- `rag_service.py`

### Result
This made the workflow integration-ready. Teammates can later connect the real model by mainly editing the service files, without changing the graph structure or node logic.

## Phase 2 — Deterministic Symptom Completion
Real-model testing showed that the model did not reliably emit `[END_SYMPTOMS]` and often kept asking questions.

### What was done
Added a dedicated `symptom_service.py` to manage:
- symptom slot extraction
- symptom slot counting
- completion rules

Tracked slots:
- location
- duration
- appearance
- severity
- triggers

Completion rules:
- `[END_SYMPTOMS]` present, or
- at least 4 slots filled, or
- maximum symptom turns reached

### Result
The workflow no longer depends only on model-generated stopping behavior and can end symptom collection deterministically.

## Phase 3 — Response Parsing and Cleanup
Real-model outputs contained noisy or malformed formatting such as:
- `<think>` blocks
- repeated `[DIAGNOSIS]`
- stray `[END_SYMPTOMS]`
- `[NOT_AVAILABLE]`
- trailing symbols

### What was done
Added a reusable parser layer in `response_parser.py` and updated `clean_response.py`.

The parser:
- removes `<think>...</think>`
- extracts the first valid `[DIAGNOSIS]...[/DIAGNOSIS]`
- strips noise tags
- removes duplicate or malformed leftovers
- returns clean user-facing text

### Result
The workflow became more tolerant of imperfect model outputs.

## Phase 4 — Diagnosis Milestone Protection
Output cleaning alone was not enough to protect the workflow from invalid backward transitions.

### What was done
Strengthened the state machine so that diagnosis is treated as a protected milestone:
- `diagnosis_done = True` is set only after a valid diagnosis response
- `phase = "follow_up"` is protected after diagnosis
- backward routing to symptom collection is blocked after diagnosis
- stale tags such as `[END_SYMPTOMS]` no longer cause regression

### Result
The workflow remains stable after diagnosis and does not accidentally return to earlier phases.

## Phase 5 — Uncertainty Handling
Model and vision outputs may be weak or conflicting.

### What was done
Added confidence-based uncertainty handling with three levels:
- high confidence: proceed normally
- medium confidence: cautious diagnosis
- low confidence: request a clearer image

The low-confidence flow was improved so that:
- symptom information is preserved
- the user is asked to upload a clearer image
- the workflow does not restart symptom collection unnecessarily

### Result
The workflow now handles uncertain image results more safely and realistically.

## Phase 6 — Backend Switching
The workflow needed to support both mock mode and future real integration mode.

### What was done
Added configuration-based switching using:
- `USE_MOCK_LLM`
- `USE_MOCK_VISION`
- `USE_MOCK_RAG`

These can be controlled from:
- `config.py`
- environment variables

### Result
The same LangGraph workflow can now run:
- with mocks for local testing
- with real services later during integration

## Phase 7 — Scenario-Based Testing
Added workflow tests to validate the orchestration systematically.

### What was done
Scenario-based tests were added for:
1. normal symptom collection
2. early image upload
3. symptom completion without `[END_SYMPTOMS]`
4. max-turn fallback
5. full diagnosis path
6. low-confidence image fallback
7. diagnosis milestone protection
8. response parsing cleanup
9. follow-up routing
10. session reset

### Result
The workflow was validated across the main normal and edge cases. During testing, I also found and fixed a real issue where `_route` was not preserved in the state schema.

## Web-Based Manual Testing
I used the local web interface to manually test the workflow and debug state behavior. This helped identify and fix issues such as:
- repeated image acknowledgement
- early-image routing loop
- symptom slot not progressing
- phase transition problems

The debug panel was useful to observe:
- phase
- symptom slots
- state flags
- nodes visited
- diagnosis status

## Final Outcome
The orchestration layer is now:
- modular
- state-driven
- robust against weak model behavior
- tolerant of noisy outputs
- configurable for mock and real backends
- tested across key workflow scenarios
- ready for teammate integration with the real model and backend services

## Conclusion
The orchestration work focused on making the multimodal dermatology chatbot workflow reliable and integration-ready. Instead of relying only on the fine-tuned model for control, strengthened the LangGraph layer to manage symptom collection, image handling, diagnosis transitions, uncertainty fallback, follow-up routing, and response cleanup. This creates a solid foundation for combining the workflow with the final real model in the next integration stage.