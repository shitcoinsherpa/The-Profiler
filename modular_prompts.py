"""
Modular sub-prompts for efficient parallel processing.
Each prompt is focused on a specific analysis aspect for reliable model following.
"""

# =============================================================================
# VISUAL ANALYSIS SUB-PROMPTS (Stage 3 - run in parallel)
# =============================================================================

VISUAL_FACS_PROMPT = """Analyze facial expressions using FACS (Facial Action Coding System).

For each distinct expression observed across the images, provide:
- Action Unit codes (e.g., AU1, AU2, AU4, AU6, AU12, AU14, AU15, AU17, AU24)
- Combination interpretation (e.g., "AU6+AU12 = Duchenne smile (genuine)")
- Timestamp/frame reference if determinable
- Emotional interpretation

Key AU Reference:
- AU1+AU2: Surprise (brow raise)
- AU4: Concentration/Anger (brow lower)
- AU6+AU12: Genuine smile (Duchenne)
- AU12 alone: Social/fake smile
- AU14: Contempt (dimpler)
- AU15: Sadness (lip corner depressor)
- AU17: Doubt (chin raiser)
- AU24: Tension (lip pressor)

Flag any AU combinations suggesting masked or incongruent emotions.
Focus ONLY on FACS coding - be precise and clinical."""

VISUAL_ARCHETYPE_PROMPT = """Identify the subject's behavioral archetype from visual evidence.

Forensic Archetype Categories:
- The Social Engineer (manipulation, charm-based influence)
- The Grievance Collector (resentment, victim mentality)
- The Pathological Liar (chronic deception, reality distortion)
- The Narcissistic Operator (grandiosity, entitlement)
- The Paranoid Vigilant (hypervigilance, distrust)
- The Impulsive Actor (low inhibition, risk-seeking)
- The Calculated Predator (methodical, strategic)
- The Unstable Reactor (emotional volatility)
- The Authority Seeker (dominance, control)
- The Chameleon (adaptive persona, identity fluidity)

Provide:
1. PRIMARY ARCHETYPE with visual evidence
2. SECONDARY ARCHETYPE if applicable
3. Confidence level (Low/Medium/High)

DO NOT compare to actors, celebrities, or fictional characters.
Focus ONLY on behavioral pattern identification."""

VISUAL_BODY_LANGUAGE_PROMPT = """Analyze body language and proxemics from the images.

Assess:
1. POSTURE: Expansion vs contraction, dominance vs submission
2. GESTURES: Emphatic, illustrative, self-soothing, barrier-creating
3. SPATIAL BEHAVIOR: Personal space, territorial markers, camera proximity
4. SELF-ADAPTORS: Face touching, grooming, object manipulation
5. ORIENTATION: Approach vs avoidance relative to camera/others

Rate each dimension 0-100:
- Dominance/Authority Projection: [score]
- Openness vs Guardedness: [score]
- Comfort/Relaxation Level: [score]
- Engagement/Interest: [score]

Provide specific visual evidence for each assessment."""

VISUAL_DECEPTION_PROMPT = """Assess deception indicators from visual evidence.

Analyze:
1. MICRO-EXPRESSIONS: Brief emotional flashes contradicting stated affect
2. GAZE PATTERNS: Avoidance, excessive contact, scanning behavior
3. FACIAL INCONGRUENCE: Mismatched upper/lower face expressions
4. SUPPRESSION MARKERS: AU24 (lip press), AU17 (chin raise), partial expressions
5. TIMING: Delayed or premature emotional displays

Provide:
- Specific deception indicators observed (with frame/time reference)
- Authenticity assessment: GENUINE / PERFORMED / MIXED
- Confidence level for deception assessment
- Hot spots requiring investigative attention

Be conservative - only flag clear indicators, note uncertainty."""


# =============================================================================
# MULTIMODAL SUB-PROMPTS (Stage 4 - run in parallel)
# =============================================================================

MULTIMODAL_GESTURE_TIMELINE_PROMPT = """Create a chronological micro-gesture timeline.

Format each entry as:
[TIMESTAMP] - [BEHAVIOR] - [INTERPRETATION] - [SIGNIFICANCE: Low/Med/High]

Log these event types:
- Micro-expressions (fleeting emotions <0.5 sec)
- Self-adaptors (face touch, hair groom, object manipulation)
- Gaze shifts (breaking contact, scanning, avoidance)
- Postural changes (leaning, shifting, limb position changes)
- Gestural clusters (multiple simultaneous tells)
- Speech-gesture mismatches

Provide at least 10-15 significant timestamped events.
Focus ONLY on creating the timeline - be precise with timestamps."""

MULTIMODAL_CROSS_MODAL_SYNC_PROMPT = """Analyze cross-modal synchronization between audio and visual channels.

For key moments, assess:
1. VOICE-FACE MATCH: Does vocal emotion match facial expression?
2. GESTURE-SPEECH TIMING: Do emphatic gestures lead or lag words?
   - Natural: Gesture precedes or coincides with emphasis
   - Rehearsed: Gesture follows after the emphasized word
3. STRESS ALIGNMENT: Do voice stress peaks match visual stress indicators?
4. TOPIC REACTIONS: What was being discussed when incongruence appeared?

Flag SYNCHRONY BREAKS as high-priority investigative points:
[TIMESTAMP] - AUDIO: [description] vs VISUAL: [description] - INTERPRETATION

Gestural-verbal asynchrony is the highest indicator of rehearsed deception."""

MULTIMODAL_ENVIRONMENT_PROMPT = """Analyze environmental and contextual elements.

Assess:
1. SETTING: Location type, staging indicators, natural vs arranged
2. OBJECTS: Luxury items (authenticity check), props, personal items
3. LIFESTYLE CONGRUENCE: Do possessions match claimed status?
4. TECHNOLOGY: Devices visible (brand, condition, generation)
5. CREDENTIALS: Certificates, awards, books visible
6. RED FLAGS: Rental items as owned, borrowed luxury, fake credentials

Note any objects that contradict verbal claims or presented identity.
Focus ONLY on environmental analysis."""

MULTIMODAL_CAMERA_AWARENESS_PROMPT = """Assess camera awareness and performance detection.

Distinguish between interpersonal deception and performative behavior:
1. Does subject glance at camera/recording device?
2. Non-Duchenne smiles - are they:
   - Duping Delight (pleasure from deceiving immediate victim)
   - Audience Wink (performance for unseen viewers)
3. Does behavior intensify when aware of recording?
4. Is there "fourth wall" awareness suggesting staged interaction?

Classification: GENUINE INTERACTION / STAGED PERFORMANCE / HYBRID

Also assess SITUATIONAL AWARENESS during disengagement:
- Scanning for security (criminal mindset)
- Checking phone/camera (creator behavior)
- Scanning exits (flight risk)
- Watching others' reactions (social engineering calibration)"""


# =============================================================================
# AUDIO SUB-PROMPTS (Stage 5 - run in parallel)
# =============================================================================

AUDIO_VOICE_CHARACTERISTICS_PROMPT = """Analyze vocal characteristics and speech patterns.

Assess:
1. PITCH: Baseline, variability, stress-induced changes
2. TEMPO: Speaking rate, pauses, hesitations, rushed segments
3. VOLUME: Projection, emphasis patterns, trailing off
4. QUALITY: Clear, raspy, breathy, strained, relaxed
5. RHYTHM: Cadence, pulsation, monotone vs dynamic

Provide baseline measurements and note deviations.
Include specific timestamps for notable vocal events."""

AUDIO_SOCIOLINGUISTIC_PROMPT = """Perform socio-linguistic origin profiling.

Analyze:
1. PRIMARY ACCENT: Regional, national, ethnic markers
2. SECONDARY INFLUENCES: Migration history, exposure patterns
3. PHONOLOGICAL MARKERS: Vowel shifts, consonant patterns, intonation
4. LEXICAL CHOICES: Regional vocabulary, idioms, slang
5. CODE-SWITCHING: Alternating dialects or registers

Estimate:
- Geographic origin (country, region, urban/rural)
- Socioeconomic markers in speech
- Education level indicators
- First language if non-native English

Be specific about linguistic evidence for each conclusion."""

AUDIO_DECEPTION_VOICE_PROMPT = """Assess vocal deception indicators.

Analyze:
1. VOCAL STRESS: Pitch changes, tension, shakiness, throat clearing
2. SPEECH DISRUPTIONS: Hesitations, corrections, false starts
3. QUALIFIERS: "To be honest," "Frankly," "I swear" - distancing language
4. TENSE SHIFTS: Past/present inconsistencies
5. DETAIL PATTERNS: Over-elaborate vs sparse in specific sections

For each indicator, provide:
- Timestamp reference
- Specific observation
- Deception probability: Low/Medium/High
- Confidence in assessment

Note areas where subject seems most/least truthful."""


# =============================================================================
# SYNTHESIS SUB-PROMPTS (Stage 6 - run in parallel, then integrate)
# These require previous analysis results injected via {previous_analyses}
# =============================================================================

SYNTHESIS_PERSONALITY_PROMPT = """Generate personality structure assessment.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

Based on the above analyses, provide:

BIG FIVE (0-100 with evidence):
- Openness to Experience: [score] | [confidence: Low/Med/High] | [evidence]
- Conscientiousness: [score] | [confidence] | [evidence]
- Extraversion: [score] | [confidence] | [evidence]
- Agreeableness: [score] | [confidence] | [evidence]
- Neuroticism: [score] | [confidence] | [evidence]

DARK TRIAD (0-100 with evidence):
- Narcissism: [score] | [confidence] | [behavioral evidence]
- Machiavellianism: [score] | [confidence] | [behavioral evidence]
- Psychopathy: [score] | [confidence] | [behavioral evidence]

MBTI HYPOTHESIS:
- Type: [4-letter code] | [confidence]
- Evidence for each dimension

Use ONLY the provided analysis data - synthesize across all modalities."""

SYNTHESIS_THREAT_PROMPT = """Generate threat assessment matrix.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

Based on the above analyses, rate 0-100 with evidence:
- Volatility Risk: [score] | [evidence]
- Manipulation Capacity: [score] | [evidence]
- Compliance Likelihood: [score] | [evidence]
- Stress Resilience: [score] | [evidence]
- Ethical Boundaries: [score] | [evidence]

THREAT LEVEL: LOW / MODERATE / HIGH / CRITICAL

Justify with:
- Risk of violence
- Risk of flight
- Risk of evidence destruction
- Cooperation likelihood
- Deception capability

Be specific and evidence-based."""

SYNTHESIS_DIFFERENTIAL_PROMPT = """Perform clinical differential diagnosis.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

Based on the above analyses, consider alternative explanations for observed behaviors:

1. MANIC/HYPOMANIC EPISODE (Bipolar): High energy, grandiosity, reduced sleep need, risky behavior, pressured speech
2. ADHD: Impulsivity, distractibility, interrupting - may mimic deception
3. AUTISM SPECTRUM: Atypical eye contact, flat affect - may be misread
4. ANXIETY DISORDERS: Nervousness, fidgeting may mimic guilt
5. SUBSTANCE INFLUENCE: Intoxication or withdrawal effects
6. PERSONALITY DISORDERS: Borderline, Histrionic, Antisocial patterns

For each relevant differential:
- Probability: Unlikely / Possible / Likely
- Supporting evidence
- How it would change interrogation approach

Flag which considerations investigators must keep in mind."""

SYNTHESIS_CONTRADICTIONS_PROMPT = """Analyze contradictions between modalities (Hot Spots).

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

Based on the above analyses, identify conflicts between:
- Visual vs Audio (confident voice but anxious eyes)
- Verbal content vs Non-verbal behavior
- Beginning vs End behavioral drift
- Self-presentation vs Micro-expression leakage

For EACH contradiction found:
1. CONFLICTING SIGNALS: [specific observations]
2. UNIFIED INTERPRETATION: Why do both exist simultaneously?
3. PSYCHOLOGICAL INSIGHT: What this reveals about strategy/psychology

Example resolution:
BAD: "Subject shows both submissive and dominant behaviors."
GOOD: "Subject employs Social Engineering - using vocal submission to appear non-threatening while maintaining physical dominance to control the interaction frame."

Contradictions ARE the profile. Resolve them, don't just list them."""

SYNTHESIS_RED_TEAM_PROMPT = """Perform Red Team analysis (self-critique).

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

Based on the above analyses, identify THREE reasons why this analysis might be WRONG:

1. ENVIRONMENTAL/CONTEXTUAL FACTORS
   What situational elements could explain behaviors differently?

2. CULTURAL/BACKGROUND CONSIDERATIONS
   What norms might be misinterpreted through Western/American lens?

3. DATA LIMITATIONS
   What blind spots exist in recording quality, duration, or analytical method?

For each, explain:
- The alternative interpretation
- How it would change the profile
- Probability this alternative is correct

This ensures investigators don't over-rely on the profile."""

SYNTHESIS_FINAL_INTEGRATION_PROMPT = """Create final integrated profile and operational recommendations.

ALL ANALYSIS DATA:
{previous_analyses}

SYNTHESIS SUB-ANALYSES:
{synthesis_results}

Integrate ALL above into the final profile:

EXECUTIVE SUMMARY:
- Subject overview (2-3 sentences)
- Primary risk classification
- Key behavioral finding

OPERATIONAL RECOMMENDATIONS:
- Optimal interview approach (rapport-based vs confrontational)
- Psychological leverage points
- De-escalation strategies
- Behaviors to monitor in real-time
- Predicted responses to various tactics

INVESTIGATIVE PRIORITIES:
- Key areas to investigate further
- Associates/relationships to examine
- Evidence sources based on personality

Synthesize ALL previous analyses into actionable intelligence.
Do NOT repeat detailed findings - reference and integrate them."""


# =============================================================================
# PROMPT GROUPS FOR PARALLEL EXECUTION
# =============================================================================

VISUAL_PROMPTS = {
    'facs': VISUAL_FACS_PROMPT,
    'archetype': VISUAL_ARCHETYPE_PROMPT,
    'body_language': VISUAL_BODY_LANGUAGE_PROMPT,
    'deception': VISUAL_DECEPTION_PROMPT,
}

MULTIMODAL_PROMPTS = {
    'gesture_timeline': MULTIMODAL_GESTURE_TIMELINE_PROMPT,
    'cross_modal_sync': MULTIMODAL_CROSS_MODAL_SYNC_PROMPT,
    'environment': MULTIMODAL_ENVIRONMENT_PROMPT,
    'camera_awareness': MULTIMODAL_CAMERA_AWARENESS_PROMPT,
}

AUDIO_PROMPTS = {
    'voice_characteristics': AUDIO_VOICE_CHARACTERISTICS_PROMPT,
    'sociolinguistic': AUDIO_SOCIOLINGUISTIC_PROMPT,
    'deception_voice': AUDIO_DECEPTION_VOICE_PROMPT,
}

SYNTHESIS_PROMPTS = {
    'personality': SYNTHESIS_PERSONALITY_PROMPT,
    'threat': SYNTHESIS_THREAT_PROMPT,
    'differential': SYNTHESIS_DIFFERENTIAL_PROMPT,
    'contradictions': SYNTHESIS_CONTRADICTIONS_PROMPT,
    'red_team': SYNTHESIS_RED_TEAM_PROMPT,
    'final': SYNTHESIS_FINAL_INTEGRATION_PROMPT,
}
