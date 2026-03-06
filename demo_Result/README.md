# DeepFace vs HORUS v0.6 — Complete Comparison Table
## For AIES 2026 Submission

---

## Table 1: Primary Emotion Classification Comparison

| Image | Subjects | Ground Truth | DeepFace | HORUS v0.6 |
|-------|----------|-------------|----------|-------------|
| demo1 (Brand store) | 2 | Relaxed joy, "watching over" dynamic | Face#1: HAPPY 95.3% ✓ / Face#2: HAPPY 96.4% ✓ | Joy 0.75-0.85 + power dynamic detected ("watching over") ✓✓ |
| demo2 (Silly selfie) | 3 | Playful joy, group fun | **1 face only detected** / FEAR 84.2% ✗✗ | 3 persons / Playfulness 0.85-0.95 ✓✓ |
| demo3 (HS students) | 7-8 | Celebratory teasing (照れ→喜び) | Face#1: **SAD 99.0%** ✗✗ / Face#2-4: HAPPY | Surface: embarrassment / Deep: joy 0.8 + mΔ=0.65 predicted ✓✓ |
| demo4 (Disney) | 1 | Excited anticipation + mild tension | NEUTRAL 81.4% ✗ | (no ctx) Social display 0.6 / (with ctx) Anticipation 0.85 + tension 0.3 ✓✓ |
| demo5 (Martial arts) | 2 | Post-training satisfaction | **3 faces detected** ✗ / Face#1: SAD 80.3% ✗✗ / Face#3: ANGRY 48.0% ✗ | 2 persons / Joy 0.9 + Reserved satisfaction 0.6 ✓✓ |

### Summary
- **DeepFace correct**: 1/5 (demo1 only — surface level)
- **DeepFace partially correct**: 0/5
- **DeepFace wrong**: 4/5 (demo2: FEAR, demo3: SAD, demo4: NEUTRAL, demo5: SAD+ANGRY)
- **HORUS correct**: 5/5 (with hierarchical depth)

---

## Table 2: Face Detection Accuracy

| Image | Actual Persons | DeepFace Detected | HORUS Detected | Notes |
|-------|---------------|-------------------|----------------|-------|
| demo1 | 2 | 2 ✓ | 2 ✓ | Both correct |
| demo2 | 3 | **1** ✗ | 3 ✓ | DeepFace missed 2 faces (duck face = undetectable) |
| demo3 | 7-8 | 4 ⚠️ | 7-8 ✓ | DeepFace missed ~half the group |
| demo4 | 1 | 1 ✓ | 1 ✓ | Both correct |
| demo5 | 2 | **3** ✗ | 2 ✓ | DeepFace hallucinated a 3rd face |

---

## Table 3: Critical Failure Analysis (DeepFace)

| Image | DeepFace Output | Actual Emotion | Error Type | Cause |
|-------|----------------|----------------|------------|-------|
| demo2 | FEAR 84.2% | Playful joy | **Complete inversion** | Furrowed brows + squinted eyes + pursed lips = "fear" features. No context integration. |
| demo3 | SAD 99.0% | Embarrassed joy (照れ) | **Complete inversion** | Downward gaze = "sadness" feature. No relationship/cultural analysis. |
| demo4 | NEUTRAL 81.4% | Excited anticipation | **Missed emotion** | Subtle tension undetectable without context. |
| demo5 Face#1 | SAD 80.3% | Post-training joy | **Complete inversion** | Unknown cause — subject is visibly smiling. |
| demo5 Face#3 | ANGRY 48.0% | Does not exist | **Hallucinated face** | Background element misdetected as face. |

---

## Table 4: HORUS Capabilities Beyond DeepFace

| Capability | DeepFace | HORUS v0.6 |
|-----------|----------|-------------|
| Single emotion label | ✓ (7 categories) | ✓ (unlimited) |
| Hierarchical emotion (Surface/Deep) | ✗ | ✓ |
| Emotion gap analysis | ✗ | ✓ |
| Relationship inference | ✗ | ✓ (Layer 3) |
| Narrative generation (アテレコ) | ✗ | ✓ (Layer 4) |
| Cultural lens integration | ✗ | ✓ (Layer 1→all) |
| mΔ prediction (misread risk) | ✗ | ✓ (Layer 7, 4-quadrant CRLM) |
| Self-bias diagnosis | ✗ | ✓ (Layer 8) |
| Context verification (anti-fake) | ✗ | ✓ (contradiction detection) |
| Evidence tagging [obs/inf-high/inf-low] | ✗ | ✓ |
| Interpretation Ranking (probability) | ✗ | ✓ |
| Photo epistemological limits | ✗ | ✓ |
| Multi-person group dynamics | ✗ | ✓ |
| Context A/B sensitivity | N/A | ✓ (demo4 experiment) |

---

## Table 5: Context Sensitivity (demo4 A/B/C Test)

| Metric | No Context | Correct Context | Fake Context |
|--------|-----------|----------------|--------------|
| Location | "Spanish restaurant" | "Disney Sea, Mediterranean Harbor" | "Spanish café" |
| Smile type | Social display 0.6 | Genuine joy 0.85 | Contradiction detected ⚠️ |
| Tension | 0.2 | 0.3 | 0.25 |
| Hand analysis | "Holding phone" | "Self-soothing, anticipation" | "Showing/presenting" |
| Top interpretation | I1: Restaurant date (P=0.55) | I1: Pre-greeting excitement (P=0.70) | I1: Fun date, context likely false (P=0.55) |
| Fake context detected | N/A | N/A | ✓ (4/5 indicators contradicted) |
| DeepFace (all conditions) | NEUTRAL 81.4% | NEUTRAL 81.4% | NEUTRAL 81.4% |

**Note**: DeepFace output is identical across all three conditions because it has no context input capability.

---

## Table 6: mΔ Predictions Across Demo Images

| Image | Literal (C₀R₀) | Cultural (C₁R₀) | Relational (C₀R₁) | Resonant (C₁R₁) |
|-------|----------------|-----------------|-------------------|-----------------|
| demo1 | 0.30 (Underdecode) | 0.15 (None) | 0.10 (None) | 0.05 (None) |
| demo2 | 0.20 (Underdecode) | 0.10 (None) | 0.10 (None) | 0.00 (None) |
| demo3 | **0.65** (Underdecode) | 0.15 (None) | 0.20 (Underdecode) | 0.05 (None) |
| demo4 fake | **0.85** (Overdecode) | 0.30 (Underdecode) | 0.50 (Mixed) | 0.15 (None) |
| demo5 | 0.30 (Overdecode) | 0.05 (None) | 0.10 (None) | 0.02 (None) |

**Key finding**: demo3 and demo4-fake show highest mΔ for Literal observers, indicating these images are most vulnerable to misinterpretation without cultural/relational context.

---

## Technical Comparison

| Metric | DeepFace | HORUS v0.6 |
|--------|----------|-------------|
| Architecture | CNN + FER2013 dataset | LLM (Claude Opus 4.5) + Prompt |
| Training data | 35,887 images | **None** |
| GPU required | Yes | **No** |
| Model size | ~6MB weights | **0** (API-based) |
| Prompt size | N/A | ~700 lines |
| Output | 7 emotion probabilities | 8-layer structured analysis |
| Context input | **Not supported** | Supported + verified |
| Cost per image | Free (local) | ~$0.05-0.10 (API) |
| Avg tokens | N/A | Input: ~5K / Output: ~5K |

---

*Generated for AIES 2026 submission: "See the Improbable Present: Misinterpretation-Aware AI Through Cultural-Relational Lens Analysis"*

*𓂀 HORUS: Hierarchical Observation & Recognition with Universal Sight*
