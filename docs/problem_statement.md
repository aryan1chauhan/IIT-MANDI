# Multimodal AI Hackathon 2026

## Track C: Contrastive Speech Analytics & Temporal Flaw Grounding

### Challenge
Develop a speech evaluation system by constructing a custom contrastive dataset of "ideal" vs. "flawed" speeches, to generate reproducible custom evaluative rubric-based scores and actionable delivery feedback via an interactive dashboard.

---

### Context
Evaluating competitive spoken performances, such as Interpretive Reading, Declamation, Extemporaneous, and Persuasive Oratory is inherently subjective. Judges must balance vocal delivery (tone, cadence, volume, pauses) with textual content quality, which produces inconsistent scoring across judges. The participants lose points for delivery flaws but receive vague, subjective feedback. Furthermore, while standard datasets exist for speech recognition, there is no definitive dataset mapping good rhetorical delivery against a spectrum of bad deliveries of the exact same text. 

To solve this, you must engineer your own contrastive dataset, sourcing prominent public speeches and mirroring them with intentionally flawed recordings ranging from completely botched to "almost perfect". By treating speech as a time-series signal and extracting features (like FFT, MFCCs etc.), the system must map acoustic stress points, temporally ground the exact region of failure, and explain the mathematical rationale behind the flaw.

---

### Requirements

#### 1. Custom Contrastive Dataset & Stress Testing
- **a. The "Good" Baseline:** Source and transcribe audio recordings of highly effective public speakers (e.g., prominent politicians, champion debaters, TED speakers).
- **b. The "Bad" Spectrum:** Synthesize, self-record, or modify audio of the exact same transcripts with intentionally injected delivery flaws, building a gradient from egregiously wrong pacing/tone to near-perfect delivery.
- **c. Labeling:** Accurately label and temporally bound paired good/bad recordings for training and testing.

#### 2. Advanced Feature Extraction & Forced Alignment
- **a. Forced Alignment:** Perform forced alignment to map the text transcript precisely to audio timestamps.
- **b. Feature Extraction:** Apply feature extraction techniques (for example, FFT, MFCCs, pitch, F0 contours, speech rate, pause intervals, vocal clarity) to capture the acoustic footprint.

#### 3. Temporal Grounding & Causal Flaw Extraction
- **a. Temporal Grounding:** Isolate exact start and end timestamps of flaw regions where delivery significantly deviates from baseline acoustic expectations.
- **b. Causal Explainability:** Translate the raw mathematical delta into a structured, human-readable causal explanation for each flaw region.

#### 4. Visualization & Dashboarding
- **a. Interactive Frontend:** Build an interactive frontend interface that accepts audio and transcript uploads.
- **b. Time-Series Overlay:** Render a time-series overlay comparing baseline and participant features, highlighting flaw regions alongside their causal explanations.

---

### Expected Deliverables
- **GitHub Repository:** Containing all codebase, tests, and documentation.
- **The Custom Dataset:** A well-documented dataset containing paired good/bad spectrum audio files, their transcripts, and alignment labels (must be in GitHub, or a public Google Drive link attached in the README).
- **Interactive Dashboard:** A functional web/app-based prototype demonstrating audio upload, feature extraction processing, and visual/textual data representation.
- **Technical Documentation:** Documentation of dataset construction, evaluation rubrics, model architecture, and scoring methodology (max. 6 pages).
- **3–10 Minute YouTube Video:** Demonstrating dataset collection, stress testing across different speech qualities, and the dashboard catching specific delivery deviations.

---

### Technical Considerations
- **Data Quality Over Quantity:** Dataset quality takes priority over quantity, favoring a cleanly aligned contrastive spectrum over a large noisy corpus.
- **Speaker Agnostic:** Comparison logic should remain speaker-agnostic, normalizing pitch and energy to account for natural biological differences between speakers.
- **Temporal Precision:** The system should identify flaw regions with sufficient temporal precision to support causal explanation.
- **Reproducibility:** The solution should remain reproducible across repeated runs and deployment environments.

---

### Evaluation Criteria
- **Causal Explainability & Temporal Grounding (25%):** Accuracy of flaw-region timestamps, clarity of mathematical rationale linking acoustic deviation to context.
- **Data Engineering & Stress Testing (30%):** Cleverness of baseline sourcing, cleanliness of temporal labeling, robustness of the bad-mirror gradient.
- **Feature Extraction (20%):** Mathematical rigor of acoustic processing, accuracy of stress-point and energy-contour identification.
- **Visualization & Dashboard (15%):** Frontend execution, usability, clarity of time-series and causal-rationale communication.
- **Reproducibility & Code Quality (10%):** Cleanliness of execution, clarity of deployment instructions, code architecture.
