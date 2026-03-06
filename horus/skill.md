---
name: multimodal-emotion-analysis
description: |
  A skill for analyzing complex emotions from images of people. Rather than single emotion labels 
  (anger, joy, etc.), it generates hierarchical emotion tensors integrating multi-person relationships, 
  environmental context, and cultural lenses.
  
  Use this skill when:
  - Asked to analyze emotions in images containing people
  - Questions like "How does this person feel?" or "What's the mood of this photo?"
  - Estimating relationships between multiple people in a photograph
  - Gap analysis between facial expressions and situational context is needed
  - Contexts such as counseling, UX research, or communication analysis
  - Both image and text (conversation or context) are provided for integrated analysis
  - Pre-posting risk checks for social media misunderstanding or controversy
  - Evaluating the adequacy of content moderation decisions
  - Predicting how an image might be misinterpreted by others
  
  Instead of simple "smile = happy" mappings, this skill infers "who" feels "what" 
  toward "whom," in "what situation," and "why" they display that expression.
  Plutchik's single-emotion classification is not used.
  
  Based on CRLM (Cultural-Relational Lens Matrix) theory, this skill predicts 
  interpretation divergence (mΔ) across observers with different lenses 
  and includes AI self-diagnosis of its own biases.
---

# Multimodal Hierarchical Emotion Analysis

## Overview

An image analysis skill that reproduces the human emotion recognition process and extends it 
to implement "social intelligence" — predicting how others with different lenses would interpret 
the same image, and diagnosing the AI's own interpretive biases.

No CNN feature extraction. No ML classifiers. Using only the LLM's multimodal reasoning 
capabilities, this skill executes the same perceptual pipeline humans use, plus two capabilities 
humans find difficult: predicting how others see the same scene, and diagnosing one's own biases.

**Core principle**: No features required. The human eye doesn't see that much. 
Define how to look — the reasoning does the rest.

**Theoretical foundation**: CRLM (Cultural-Relational Lens Matrix)
— The same image yields different interpretations depending on the observer's lens (cultural and relational).
This skill predicts that interpretation divergence (mΔ) and enables AI self-diagnosis of bias.

## Processing Pipeline

Upon receiving an image, execute the following 8-layer pipeline **strictly in this order**.
Each layer's output feeds into the next. No layer may be skipped.

**Pipeline overview:**
- Layers 1–4: Input processing (understanding what is happening)
- Layers 5–6: Analysis and integration (inferring emotions and intentions)
- Layers 7–8: Metacognition (evaluating misinterpretation risk and self-bias)

### Layer 1: Wide-Angle Attention — Overall Context

Survey the entire image to grasp the **scene context**. Do not yet analyze individual people.

**Elements to identify:**
- Location (indoor/outdoor, specific setting: living room, office, park, etc.)
- Estimated time of day (from lighting and light direction)
- Number of people and general arrangement (distances, orientations)
- Clothing and appearance (formal/casual, uniformity, cultural cues)
- Surrounding objects and environmental elements (food, tools, decorations, etc.)
- Overall atmosphere (tense/relaxed, formal/casual)

**Output format:**
```
[Layer 1: Overall Context]
- Location: {estimated location}
- Time of day: {estimate}
- People: {N people}
- Arrangement: {distances and orientations}
- Clothing: {summary}
- Environmental elements: {notable items}
- Overall atmosphere: {one-line summary}
```

### Layer 2: Narrow-Angle Attention — Individual Analysis

**While retaining** the Layer 1 context, analyze each person individually.
Never analyze facial expressions without context.

**Elements to assess for each person:**
- Eyes: degree of openness, gaze direction (who/what they are looking at), shape of outer corners
- Eyebrows: angle, degree of furrowing, tension
- Mouth: shape (open/closed, direction of corners), tension/relaxation
- Face orientation: frontal/profile/downward, tilt
- Posture: body openness, lean, arm position
- Hands: what they are doing, where they are
- Overall physical tension: relaxed/tense assessment

**Output format (per person):**
```
[Layer 2: Person {N} — Individual Analysis]
- Eyes: {analysis}
- Eyebrows: {analysis}
- Mouth: {analysis}
- Face orientation: {analysis}
- Posture: {analysis}
- Hands: {analysis}
- Physical tension: {0.0–1.0}
```

### Layer 3: Relationship Inference

**Integrate** Layer 1 (context) and Layer 2 (individual analysis) to infer 
interpersonal relationships.

**Elements to infer:**
- Physical distance between people and its meaning (indicator of intimacy)
- Gaze intersection patterns (who is looking at whom)
- Body orientation toward each other (open/closed)
- Power dynamics (presence/absence of hierarchy)
- Estimated intimacy: stranger / acquaintance / friend / intimate / family
- Interaction type: cooperation / conflict / playful teasing / indifference

**Important:**
- An "angry" expression with close distance and the other person laughing → possible playful teasing
- A smile with far distance and no eye contact → possible social performance
- Never judge from one person's expression alone. Interpret expressions within relationships.

**Output format:**
```
[Layer 3: Relationship Inference]
- Interpersonal distance: {close/moderate/far} → estimated intimacy
- Gaze patterns: {who is looking at whom}
- Power dynamics: {equal / hierarchical / protector-protected}
- Interaction type: {type}
- Relationship estimate: {specific estimate}
```

### Layer 4: Narrativization (Voice-Over)

**This is the most critical layer.** Integrate all information from Layers 1–3 to 
generate a **narrative** of "what is happening in this scene."

Human emotion recognition does not directly map expressions to emotions. 
It first constructs a "story," then infers emotions from that story. 
This layer reproduces that process.

**Procedure:**
1. Integrate information from Layers 1–3
2. Estimate "what happened just before this moment"
3. Estimate what each person is "feeling and trying to do"
4. Where possible, generate each person's inner voice as a **voice-over**

**Output format:**
```
[Layer 4: Narrativization]
- Scene estimate: {what is happening}
- Preceding event (estimated): {estimate}
- Voice-overs:
  - Person 1: "{inner voice}"
  - Person 2: "{inner voice}"
```

### Layer 5: Hierarchical Emotion Inference

Based on the Layer 4 narrative, infer each person's emotions in a **hierarchical structure**.
Do not use Plutchik's single-emotion labels.

**Emotion hierarchy:**
- **Surface Emotion (Parent)**: Externally observable surface emotion. Rate 0.0–1.0.
- **Deep Emotion (Child)**: Emotion hidden beneath the surface emotion. Rate 0.0–1.0.
- **Emotion direction**: Toward whom/what
- **Emotion function**: Why this emotion is being expressed (defense, expression, suppression, performance, etc.)

**Important rules:**
- Surface and deep emotions **may contradict** (anger hiding love, smiles hiding anxiety)
- When contradiction exists, perform **gap analysis**
- A single person may have multiple surface emotions

**Output format (per person):**
```
[Layer 5: Person {N} — Hierarchical Emotions]
- Surface emotion: {emotion} ({0.0–1.0}), {emotion} ({0.0–1.0})
- Deep emotion: {emotion} ({0.0–1.0}), {emotion} ({0.0–1.0})
- Emotion direction: {toward whom/what}
- Emotion function: {why this emotion is being expressed}
- Gap: {analysis of surface-deep mismatch, if present}
```

### Layer 6: Integrated Report

Integrate results from all layers into a comprehensive analysis report.

**Output format:**
```
[Emotion Analysis Report]

■ Scene Overview
{Integrated summary of Layer 1 + Layer 4}

■ Person Analysis
  Person 1:
    Expression impression: {Layer 2 summary}
    Estimated emotions:
      Surface: {emotion} ({value})
      Deep: {emotion} ({value})
    Voice-over: "{inner voice}"

  Person 2:
    (same structure)

■ Relationships
{Layer 3 summary}

■ Gap Analysis (if applicable)
{Analysis of surface-deep mismatches}

■ Cultural/Situational Notes (if applicable)
{Dialect, cultural differences, special context}
```

### Layer 7: mΔ Prediction — Interpretation Divergence Risk Assessment

**Based on CRLM (Cultural-Relational Lens Matrix): predicting misinterpretation risk per observer type.**

Predict: "If someone other than me saw this image, how might they misinterpret it?"
Analyze how the gap between surface emotion (Parent) and deep emotion (Child) detected in 
Layer 5 would cause misreadings by observers with different lenses.

**CRLM 4-Quadrant Observer Profiles:**

| Quadrant | Lens | Example Observer |
|----------|------|-----------------|
| Literal (C₀R₀) | No Cultural × No Relational | Complete outsider, AI auto-filter, random social media viewer |
| Cultural (C₁R₀) | Cultural × No Relational | Same cultural background but not personally acquainted |
| Relational (C₀R₁) | No Cultural × Relational | Different culture but personally knows the people |
| Resonant (C₁R₁) | Cultural × Relational | Same culture and personally knows the people |

**For each profile, infer:**
1. `perceived_act` — the act this observer would perceive
2. `actual_act` — the actual act as inferred in Layers 4–5
3. `mΔ` — divergence between perceived and actual (0.0–1.0)
4. Error type classification

**Error type definitions:**
- **Underdecode (Under-reading)**: Misses the deep intention. Sees only the surface and judges "no problem," but there is actually a hidden SOS or concealed hostility → **Invisible mΔ (invisible divergence)**
- **Overdecode (Over-reading)**: Reads depth that doesn't exist. Projects malice or sexual intent onto mere social custom or cultural practice → **Phantom mΔ (phantom divergence)**

**Output format:**
```
[Layer 7: mΔ Prediction (Interpretation Divergence Analysis)]

■ Per-Observer Profile Predictions

  Literal (C₀R₀) — Outsider / Auto-filter:
    Perceived act: {perceived_act}
    Actual act:    {actual_act}
    mΔ: {0.0–1.0}
    Error type: {Overdecode / Underdecode / None}
    Specific risk: {what would be misunderstood and how}

  Cultural (C₁R₀) — Same-culture third party:
    Perceived act: {perceived_act}
    Actual act:    {actual_act}
    mΔ: {0.0–1.0}
    Error type: {Overdecode / Underdecode / None}
    Specific risk: {what would be misunderstood and how}

  Relational (C₀R₁) — Cross-cultural acquaintance:
    Perceived act: {perceived_act}
    Actual act:    {actual_act}
    mΔ: {0.0–1.0}
    Error type: {Overdecode / Underdecode / None}
    Specific risk: {what would be misunderstood and how}

  Resonant (C₁R₁) — Same-culture close acquaintance:
    Perceived act: {perceived_act}
    Actual act:    {actual_act}
    mΔ: {0.0–1.0}
    Error type: {Overdecode / Underdecode / None}

■ Overall Risk Assessment
  Overdecode risk:  {High/Medium/Low} — {reason}
  Underdecode risk: {High/Medium/Low} — {reason}
  
■ Social Media Posting Simulation (if applicable)
  If this image were posted on social media:
  - Most likely misinterpretation: {specific misinterpretation}
  - Controversy risk: {High/Medium/Low}
  - Recommended action: {add caption / crop / do not post / etc.}
```

### Layer 8: Self-Diagnosis — AI Lens Bias Evaluation

**The most critical metacognitive layer.** Evaluate which lens this AI itself 
is using to interpret this image.

The question is not "Is my judgment correct?" but rather 
"What biases am I bringing to this judgment?"

**Self-diagnosis items:**

1. **Current reading depth**: Which CRLM quadrant am I reading from?
   - Literal: Am I judging only from surface features?
   - Cultural: Am I importing a specific cultural bias?
   - Relational: Am I projecting a relationship that doesn't exist?

2. **Shape bias**: In judging physical features, am I applying 
   socially constructed categories (masculine/feminine) without verification?

3. **Context dependency**: Am I appropriately weighting Layer 1 environmental 
   information? Am I rushing to judgment from Layer 2 physical features alone 
   without environmental context?

4. **Cultural bias**: Am I implicitly applying a specific cultural value system 
   as "default"?
   (e.g., Am I assuming Western standards of body exposure as universal?)

5. **Overdecode/Underdecode self-detection**:
   - Signs of self-Overdecode: Is my Layer 4 narrative excessively interpretive 
     relative to the observational facts in Layer 2?
   - Signs of self-Underdecode: Do I possess cultural or relational context 
     but fail to apply it, defaulting to Literal judgment?

**Output format:**
```
[Layer 8: Self-Diagnosis (AI Lens Bias Evaluation)]

■ Current Reading Depth
  Assessment: {Literal / Cultural / Relational / Resonant}
  Reason: {why reading at this depth}

■ Bias Detection
  Shape bias: {detected / not detected} — {details}
  Cultural bias: {detected / not detected} — {details}
  Context dependency: {appropriate / insufficient / excessive} — {details}

■ Overdecode/Underdecode Self-Check
  Signs of Overdecode: {present / absent} — {specifically where}
  Signs of Underdecode: {present / absent} — {specifically where}

■ Confidence Assessment
  Confidence in this analysis: {High / Medium / Low}
  Factors reducing confidence: {insufficient information / cultural knowledge gaps / ambiguous expressions / etc.}
  Points that would improve with additional information: {what would increase accuracy}
```

## Special Case Handling

### When Text Information (Context) Is Provided

**Critical principle: Context is not necessarily truthful.**

The human providing context also has a lens. That lens may introduce 
misunderstanding, prejudice, or even deliberate falsehood into the context.

**Do not take context at face value.**

When text information is provided alongside the image, follow this procedure:

1. First execute Layers 1–2 **without context** to establish observational facts from the image
2. Then read the context and cross-reference with Layer 1–2 observational facts
3. If context and observations are consistent → integrate context from Layer 3 onward
4. If context and observations contradict → **explicitly report the contradiction**, 
   prioritize observational facts while presenting both interpretations

**Examples of contradiction detection:**
- Context says "during a fight" but everyone is smiling with low physical tension → contradiction
- Context says "sad scene" but eye corners show Duchenne smile characteristics → contradiction
- Context says "first meeting" but physical distance is zero with natural close contact → contradiction

**Additional Layer 8 diagnosis when context is provided:**
Verify whether the provided context exhibits:
- The provider's Overdecode (projecting nonexistent malice or intent)
- The provider's Underdecode (overlooking important contextual factors)
- Possible deliberate falsehood (judged by degree of image-context inconsistency)

When spoken dialogue text is provided, perform additional **speech-expression gap analysis**:

- Speech content matches expression: No gap → surface emotion reflects depth
- Speech content contradicts expression: Gap present → infer suppression, performance, embarrassment, etc.
- Person not speaking: Infer the meaning of silence from context

### When Persona Information Is Provided

When a person's persona (personality traits, cultural background, relationship information) 
is provided, reflect it in Layer 5 emotion inference.

**The same expression yields different emotion inferences depending on persona.**
This is not a bug — it is by design. Human emotion recognition also changes depending on 
whether we know the person. However, persona information, like context, may be filtered through 
the provider's lens, so verify consistency with image observational facts.

### Single-Person Images

When only one person is present, Layer 3 (Relationship Inference) may be abbreviated.
However, Layer 1 (Environmental Context) must always be executed.
The environment changes the interpretation of emotion 
(a smile in an office vs. a smile in a bedroom).

## Prohibited Actions

- **Do not determine emotions from expressions alone**: Always integrate with context
- **Do not force-fit into Plutchik's 8 emotions**: Allow compound and contradictory emotions
- **Do not end with a single emotion label**: Always output hierarchical structure (surface/deep)
- **Do not ignore relationships**: For multiple people, do not complete analysis with individual assessment alone
- **Do not overlook cultural context**: Estimate cultural background from clothing, location, physical distance
- **Do not skip narrativization**: Layer 4 is the most critical layer determining emotion inference accuracy
- **Do not skip mΔ prediction**: Layer 7 is the social intelligence layer predicting "how others would misread this." Even if the emotion analysis is correct, failing to predict others' misinterpretations halves the practical value
- **Do not skip self-diagnosis**: Layer 8 is the metacognitive layer evaluating one's own biases. Without asking "does my own lens carry bias?" the AI cannot self-detect Overdecode/Underdecode
- **Do not casually conclude "no bias" in self-diagnosis**: Some bias is always present. "Not detected" and "does not exist" are different
- **Do not take provided context at face value**: Context is filtered through the provider's lens. When it contradicts image observational facts (Layers 1–2), prioritize observational facts. Always explicitly report contradictions when found
