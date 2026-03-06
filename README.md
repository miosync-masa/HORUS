# 𓂀 HORUS

**Hierarchical Observation & Recognition with Universal Sight**

CRLM-based 8-Layer Multimodal Emotion Analysis Pipeline

> "No features required. The human eye doesn't see that much. Define how to look — the reasoning does the rest. "

---

## What is HORUS?

HORUS is a multimodal emotion analysis system that replaces feature extraction with structured reasoning. No CNN. No training data. No GPU. Just a prompt that teaches an LLM *how to see*.

Traditional emotion recognition:
```
Image → CNN → Feature extraction → Classifier → "Anger: 0.7"
```

HORUS:
```
Image → LLM + 8-Layer Pipeline → Hierarchical emotion tensor
        with cultural lens analysis, mΔ prediction,
        and AI self-bias diagnosis
```

### Why?

A frowning face in a group selfie is **joy**, not anger.  
A smiling crowd around one person might be **bullying**, not fun.  
The same smile is **genuine** at Disneyland but **social display** at a restaurant.

Feature-based models can't distinguish these. HORUS can — because it *sees* context, relationships, and cultural lenses before interpreting expressions.

---

## Theoretical Foundation

HORUS implements the **CRLM (Cultural-Relational Lens Matrix)** framework:

```
mΔ ≡ divergence(act_A, act_B)
where act_i = f(image, lens_i, context)
```

The same image generates different interpretations through different cultural and relational lenses. HORUS predicts these divergences and diagnoses its own lens biases.

| Quadrant | Cultural Lens | Relational Lens | Example |
|----------|:---:|:---:|---------|
| **Literal** (C₀R₀) | ✗ | ✗ | AI filter, stranger on SNS |
| **Cultural** (C₁R₀) | ✓ | ✗ | Same culture, doesn't know the person |
| **Relational** (C₀R₁) | ✗ | ✓ | Different culture, knows the person |
| **Resonant** (C₁R₁) | ✓ | ✓ | Same culture, close relationship |

---

## 8-Layer Pipeline

```
Layer 1  Wide Attention      — Scene context (where, when, who)
Layer 2  Narrow Attention    — Individual analysis (eyes, mouth, posture)
Layer 3  Relationship        — Interpersonal dynamics
Layer 4  Storylization       — Narrative generation ("What's happening?")
Layer 5  Hierarchical Emotion — Parent/Child emotion with gap analysis
Layer 6  Integration Report  — Comprehensive summary
Layer 7  mΔ Prediction       — How would others misread this? (CRLM)
Layer 8  Self-Diagnosis      — AI's own lens bias evaluation
```

### Key Design Principles

- **Layer 1 determines everything.** Context selection cascades through all subsequent layers.
- **Layer 4 (Storylization) is the most critical layer.** Humans don't map expressions to emotions directly — they construct a story first, then infer emotions from the story.
- **Layer 7 predicts misunderstanding before it happens.** SNS posting risk, Overdecode/Underdecode detection.
- **Layer 8 never concludes "no bias".** Some bias always exists. "Not detected" ≠ "not present."
- **Context is not trusted blindly.** Provided context may contain the provider's own lens biases or intentional falsehoods. Image observations (Layer 1-2) take priority over contradictory context.

---

## Installation

```bash
pip install anthropic python-dotenv
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
CLAUDE_MODEL=claude-opus-4-5-20250924
```

---

## Usage

```bash
# Full 8-layer pipeline (default: budget=15000, output=result.json)
python emotion_analysis.py --image photo.jpg

# With context
python emotion_analysis.py --image photo.jpg \
  --context "After a graduation ceremony"

# With persona information
python emotion_analysis.py --image photo.jpg \
  --persona "Introverted personality, rarely smiles in public"

# Lightweight mode (Layer 1-6 only, skip mΔ and self-diagnosis)
python emotion_analysis.py --image photo.jpg --layers 1-6

# Show Extended Thinking process
python emotion_analysis.py --image photo.jpg --show-thinking

# Custom output path
python emotion_analysis.py --image photo.jpg --output analysis/demo1.json
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image`, `-i` | (required) | Image file path |
| `--context`, `-c` | None | Additional context (situation, background) |
| `--persona`, `-p` | None | Person's personality/cultural background |
| `--layers`, `-l` | `1-8` | Layers to execute (e.g., `1-6`, `1,2,5,7`) |
| `--model`, `-m` | `.env` or `claude-opus-4-5` | Model to use |
| `--budget`, `-b` | `15000` | Extended Thinking budget tokens |
| `--output`, `-o` | `result.json` | Output JSON file path |
| `--show-thinking` | False | Display thinking process |
| `--retries` | `3` | Retry count on 529 errors |

---

## Demo Results

### Demo 1: High School Students Playing

A group of Japanese high school boys roughhousing in a gymnasium.

| Observer Profile | Perceived Act | mΔ | Error Type |
|-----------------|---------------|-----|------------|
| Literal (C₀R₀) | "Bullying" | **0.85** | Overdecode |
| Cultural (C₁R₀) | "Boys being boys" | 0.15 | None |
| Resonant (C₁R₁) | "The usual" | 0.05 | None |

**Feature-based model output:** "Joy: 0.9" — completely misses the nuance.  
**HORUS output:** Detects the Overdecode risk, warns about SNS posting danger, and self-diagnoses its own "anti-bullying training data" bias.

### Demo 2: Silly Faces Selfie

Three girls making duck faces with peace signs.

| Observer Profile | Perceived Act | mΔ | Error Type |
|-----------------|---------------|-----|------------|
| Literal (C₀R₀) | "Making weird faces" | 0.45 | Underdecode |
| Cultural (C₁R₀) | "Typical selfie culture" | 0.05 | None |

**Feature-based model output:** "Anger: 0.7" — frowning = anger. **Completely wrong.**  
**HORUS output:** Correctly identifies joy through context (selfie pose, peace signs, group proximity). Also detects that the youngest child's eyes are open while others are closed — a sign of imitation learning.

### Demo 4: Context A/B Test

Same image analyzed with and without context ("Before Disney Sea character greeting").

| Metric | No Context | With Context |
|--------|-----------|--------------|
| Location | "Spanish restaurant" | "Disney Sea, Mediterranean Harbor" |
| Tension | 0.2 | 0.3 |
| Smile type | "Social display (0.6)" | "Genuine joy (0.85)" |
| Hand analysis | "Holding phone" | "Self-soothing, anticipation" |

**Context changes not just emotion intensity, but emotion TYPE and AUTHENTICITY judgment.**

---

## Architecture

```
┌─────────────────────────────┐
│  emotion_analysis.py (CLI)  │
├─────────────────────────────┤
│  .env (API key, model)      │
├─────────────────────────────┤
│  SKILL.md (8-Layer Pipeline │
│  Definition — embedded in   │
│  system prompt)             │
├─────────────────────────────┤
│  Claude Opus 4.5 API        │
│  + Extended Thinking         │
│  + Multimodal Vision         │
└─────────────────────────────┘
```

**Total implementation: ~1000 lines (Python + SKILL.md)**  
**Training data: 0**  
**GPU required: 0**  
**Feature extraction: 0**

---

## Roadmap

- [x] Phase 1: Core 8-Layer Pipeline (Layer 1-8)
- [x] Phase 1.1: mΔ Prediction (Layer 7) + Self-Diagnosis (Layer 8)
- [x] Phase 1.2: Context verification (anti-fake defense)
- [ ] Phase 2: Claude 4.6 Adaptive Thinking support
- [ ] Phase 2.1: Web Search integration (Fact Grounding)
- [ ] Phase 3: Reverse image search API integration
- [ ] Phase 4: Video frame analysis (temporal emotion tracking)
- [ ] Phase 5: API service (Lambda + API Gateway)

---

## Relation to Other Projects

HORUS is part of a unified research program:

| Project | Domain | Core Idea |
|---------|--------|-----------|
| **δ-theory** | Materials Science | Discrete structure → continuous behavior |
| **Λ³** | Physics | Hierarchical constraint satisfaction |
| **Meaning Between Lenses** | Linguistics | mΔ = divergence between cultural lenses |
| **TAP** | Translation | Persona-aware affective translation |
| **divergence-z** | Character AI | Structural persona → emergent consciousness |
| **HORUS** | Emotion Recognition | "How to see" → LLM reasoning does the rest |

All share the same principle: **Define the structure correctly, and behavior emerges.**

---

## License

MIT

---

## Citation

If you use HORUS in your research, please cite:

```bibtex
@software{horus2026,
  title={HORUS: Hierarchical Observation \& Recognition with Universal Sight},
  author={Iizumi, Masamichi},
  year={2026},
  url={https://github.com/miosync-masa/horus}
}
```

---

## Related Papers

- *Meaning Between Lenses: The Action-Theoretic Turn in Semantic Evolution* (submitted to JLE)
- *TAP: Translation with Affective Personas* (submitted to JAT)

---

*𓂀 The Eye sees not what is shown, but what the lens reveals.*
