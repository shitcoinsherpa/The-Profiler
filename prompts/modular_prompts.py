"""
Modular sub-prompts for efficient parallel processing.
Each prompt is focused on a specific analysis aspect for reliable model following.
"""

# =============================================================================
# VISUAL ANALYSIS SUB-PROMPTS (Stage 3 - run in parallel)
# =============================================================================

VISUAL_FACS_PROMPT = """Analyze facial expressions using FACS (Facial Action Coding System).

For each distinct expression observed in the video, provide:
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

VISUAL_BODY_LANGUAGE_PROMPT = """Analyze body language and proxemics in the video.

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

VISUAL_CONGRUENCE_PROMPT = """Assess CONGRUENCE between verbal content and visual behavior (non-judgmental credibility analysis).

NOTE: This is a CONGRUENCE analysis, not a "deception detection." The goal is to identify
alignment or misalignment between channels - NOT to conclude someone is "lying."
Incongruence may indicate: stress, discomfort, performance anxiety, cultural differences,
exaggeration for effect, or yes, possible deception. CONTEXT DETERMINES MEANING.

DURATION CLASSIFICATION (Critical for authenticity):
- MICRO-EXPRESSION: <0.5 seconds = Emotional LEAKAGE (genuine, suppressed)
- NORMAL EXPRESSION: 0.5-4 seconds = Typical genuine emotion
- FROZEN EXPRESSION: >4-5 seconds = PERFORMED emotion (may be theatrical, not deceptive)

Analyze:
1. MICRO-EXPRESSIONS: Brief emotional flashes contradicting stated affect
   - Duration (onset-apex-offset timing)
   - Classify: Leakage vs Suppression vs Normal
2. GAZE PATTERNS: Avoidance, excessive contact, scanning behavior
3. FACIAL INCONGRUENCE: Mismatched upper/lower face expressions
4. SUPPRESSION MARKERS: AU24 (lip press), AU17 (chin raise), partial expressions
5. TIMING ANALYSIS:
   - Expression ONSET: Before, during, or after verbal statement?
   - Expression DURATION: <0.5s / 0.5-4s / >4s
   - Natural emotions peak and fade; performed ones appear suddenly and hold

6. OCULAR ACCESS CUES (NLP Eye Patterns):
   Track eye movement direction relative to verbal content:
   - Up-Right: Visual Constructed (imagining - could be creative recall OR fabrication)
   - Up-Left: Visual Remembered (recalling actual memory)
   - Lateral-Right: Auditory Constructed (imagining sounds/dialogue)
   - Lateral-Left: Auditory Remembered (recalling actual conversation)
   - Down-Right: Kinesthetic (accessing feelings)
   - Down-Left: Internal Dialogue (self-talk)

   NOTE: Visual Constructed during factual claims = INCONGRUENCE (investigate, don't conclude)

7. BASELINE-TO-STIMULUS DELTA SCORING (0-10):
   For each topic/stimulus discussed, rate:
   - BASELINE INTENSITY (0-10): Subject's calm/neutral state
   - STIMULUS REACTION (0-10): Intensity during specific topic
   - DELTA: [Stimulus - Baseline] = Change score

   Example: "Topic: Financial losses. Baseline: 3 -> Stimulus: 8 = Delta +5 (HIGH REACTIVITY)"

   Deltas:
   - 0-2: Normal variation
   - 3-4: Moderate reactivity (notable)
   - 5-7: High reactivity (investigate)
   - 8-10: Extreme reactivity (critical flag)

Provide:
- CONGRUENCE ASSESSMENT: ALIGNED / PARTIALLY ALIGNED / MISALIGNED
- Specific incongruence markers with DURATION and timestamp
- Expression duration classification for each emotional display
- Ocular access patterns (descriptive, not accusatory)
- DELTA SCORES for top 3-5 reactive topics
- Hot spots requiring investigative attention
- ALTERNATIVE EXPLANATIONS for each incongruence (stress, performance, cultural, etc.)

Be conservative - identify incongruence, do not diagnose deception. Note uncertainty."""


# =============================================================================
# NCI/CHASE HUGHES VISUAL SUB-PROMPTS (Stage 3 - run in parallel)
# Based on The Behavior Ops Manual and Six-Minute X-Ray methodologies
# =============================================================================

VISUAL_BLINK_RATE_PROMPT = """Interpret blink rate patterns as stress and deception indicators.

**CRITICAL: CV-DETECTED BLINK DATA (GROUND TRUTH)**
{cv_blink_data}

The above data was measured by computer vision using MediaPipe Face Mesh EAR
(Eye Aspect Ratio) algorithm. These are ACCURATE measurements of actual eye closures.
DO NOT estimate your own blink rates - USE THE CV DATA ABOVE.

YOUR ROLE: INTERPRET the CV data, don't detect blinks yourself.
- The CV has already counted the blinks accurately
- Your job is to CORRELATE blink patterns with CONTENT/TOPICS
- Explain WHY stress spikes occurred at specific moments
- Connect the physiological data to what was being discussed

BASELINE REFERENCE (Chase Hughes/NCI methodology):
- Normal baseline: 17-25 blinks per minute (BPM)
- Focused/calm state: 7-10 BPM (reduced blinking indicates engagement)
- Stressed/deceptive state: Up to 50 BPM (elevated blinking indicates stress)

ANALYSIS TASKS (using CV data above):

1. STRESS WINDOW INTERPRETATION:
   - Look at the STRESS WINDOWS from CV data
   - What was the subject discussing during those timestamps?
   - Why might these topics have triggered elevated blinking?

2. BLINK CLUSTER INTERPRETATION:
   - Look at the BLINK CLUSTERS from CV data
   - These are rapid sequences of blinks in short time windows
   - What topic/memory was being accessed when clusters occurred?

3. TOPIC-CORRELATED ANALYSIS:
   - Match CV timestamps to video content
   - Which topics correlated with elevated blinking?
   - Which topics showed the calm baseline rate?

4. BEHAVIORAL SIGNIFICANCE:
   - Using the CV baseline vs peak delta, assess stress level
   - Elevated blink rate alone does NOT prove deception
   - Note as STRESS INDICATOR requiring follow-up

Provide (USING CV DATA, not your own estimates):
- CV-measured baseline: [from data above] BPM
- CV-measured peak: [from data above] BPM
- CV-measured delta: [from data above] %
- Delta Classification:
  - <50% increase: Normal variation
  - 50-150% increase: Moderate stress
  - 150-300% increase: HIGH stress
  - >300% increase: EXTREME stress
- TOPIC CORRELATION: What was discussed during stress windows?
- CLUSTER INTERPRETATION: Why did blink clusters occur at those moments?
- Stress correlation confidence: Low/Medium/High

TIMELINE FORMAT (correlating CV data with content):
[CV Timestamp] - [CV BPM] - [Delta] - [What subject was discussing]

Focus on INTERPRETATION and CORRELATION - the CV has already done the detection."""


VISUAL_BTE_SCORING_PROMPT = """Score behavioral indicators using the Behavioral Table of Elements (BTE) methodology.

CHASE HUGHES BTE SCORING SYSTEM:
Score individual stress/deception indicators. A cumulative score of 12+ suggests likely deception.

CRITICAL BASELINE CONSIDERATION:
- Self-adaptors (face touching, nose touching, grooming) are ONLY meaningful when compared to baseline
- A subject who frequently touches their face at baseline should NOT be scored highly for face touching
- The "Pinocchio Effect" (nose touching due to erectile tissue engorgement) is real but requires baseline
- Only score indicators that represent CHANGE from comfortable baseline behavior
- Without baseline, weight these indicators LOWER than other BTE items

Score each observed indicator (0-3 points based on intensity):

FACIAL INDICATORS:
- Lip compression/tension (AU24): [0-3]
- Chin raise/doubt (AU17): [0-3]
- Nostril flare: [0-3]
- Brow furrow (AU4): [0-3]
- Eye narrowing (AU7): [0-3]
- Forced/asymmetric smile: [0-3]

BODY INDICATORS:
- Single-shoulder shrug (HIGH deception indicator): [0-3]
- Foot withdrawal (pulling feet under chair): [0-3]
- Elbow closure (elbows moving inward): [0-3]
- Arm closure (arms moving toward body): [0-3]
- Postural shift backward: [0-3]
- Self-adaptors (face touching, grooming): [0-3]

CONCEALMENT CLUSTER (multiple = HIGH alert):
- Throat clasping/touching: [0-3]
- Lips pulled back: [0-3]
- Neck muscle tension: [0-3]
- Widened eyes: [0-3]

Provide:
1. INDIVIDUAL SCORES for each observed indicator
2. CUMULATIVE BTE SCORE: [total]
3. THRESHOLD ASSESSMENT:
   - Below 8: Low deception probability
   - 8-12: Moderate - requires attention
   - 12+: High deception probability
4. HIGHEST SCORING MOMENTS: [timestamps/contexts]
5. PRIMARY INDICATORS: List top 3 most prominent indicators
6. COGNITIVE LOAD TIMELINE:
   Map cognitive load intensity across the recording:
   - [MM:SS]-[MM:SS]: LOW/MEDIUM/HIGH cognitive load - [context/topic]
   - [MM:SS]-[MM:SS]: LOW/MEDIUM/HIGH cognitive load - [context/topic]
   (Continue for each distinct segment. This enables timeline visualization.)

Focus ONLY on BTE scoring - be systematic and precise."""


VISUAL_FACIAL_ETCHING_PROMPT = """Analyze facial etching patterns for long-term personality indicators.

CHASE HUGHES FACIAL ETCHING THEORY:
Repeated emotional expressions leave permanent marks on the face, revealing habitual emotional states.

Analyze visible facial etching patterns:

1. CROW'S FEET (outer eye corners):
   - Presence: Yes/No
   - Depth: Light/Moderate/Deep
   - Interpretation: Frequent genuine smiling, happiness, social warmth
   - Personality indication: Positive affect, agreeableness

2. FOREHEAD LINES (horizontal):
   - Presence: Yes/No
   - Pattern: Single/Multiple/Deep
   - Interpretation: Frequent surprise, social engagement, expressiveness
   - Personality indication: Openness, social attentiveness

3. GLABELLA LINES (between eyebrows - "11" lines):
   - Presence: Yes/No
   - Depth: Light/Moderate/Deep
   - Interpretation: Chronic anger, frustration, concentration, or depression
   - Personality indication: Neuroticism, aggression, or analytical focus

4. NASOLABIAL FOLDS (nose to mouth):
   - Presence: Yes/No
   - Depth: Light/Moderate/Deep
   - Interpretation: Frequent smiling or expressions of disgust
   - Distinguish: Smile-etched vs. contempt-etched

5. MARIONETTE LINES (mouth to chin):
   - Presence: Yes/No
   - Depth: Light/Moderate/Deep
   - Interpretation: Chronic frowning, sadness, or disapproval
   - Personality indication: Negative affect, depression markers

6. OVERALL ETCHING PATTERN:
   - Dominant emotional history: POSITIVE / NEGATIVE / MIXED
   - Age-congruence: Do lines match apparent age?
   - Asymmetry: Note any asymmetric etching (may indicate masked emotions)

Provide:
- Facial etching profile summary
- Inferred habitual emotional state
- Personality implications
- Congruence with current presentation (does current behavior match etched history?)

Focus ONLY on facial etching analysis."""


VISUAL_GESTURAL_MISMATCH_PROMPT = """Detect gestural mismatches and timing asynchrony as CONGRUENCE indicators.

CHASE HUGHES GESTURAL-VERBAL SYNCHRONY ANALYSIS:
Gestural-verbal asynchrony is one of the highest indicators of rehearsed content.

**PRECISION LIMITATION WARNING:**
LLMs cannot measure millisecond-level timing with high accuracy from video.
For claims about gesture-speech timing:
- Be CONSERVATIVE in timing claims
- Use qualitative descriptors: "gesture appears to precede/follow/coincide"
- Do NOT claim specific millisecond measurements without CV backing
- Flag high-confidence vs low-confidence observations
- Note: CV-backed timing data from MediaPipe would validate these observations

Analyze for these specific mismatches:

1. GESTURAL-VERBAL TIMING (QUALITATIVE - not millisecond-precise):
   - Natural speech: Gesture PRECEDES or COINCIDES with verbal emphasis
   - Rehearsed: Gesture FOLLOWS after the emphasized word
   - Example: Hand chop on "opportunity" - does gesture appear BEFORE/DURING or AFTER?
   - CONFIDENCE: Rate each observation as HIGH/MEDIUM/LOW confidence

2. HEAD SHAKE MISMATCH:
   - Subject says positive statement while shaking head "no"
   - Subject says negative statement while nodding "yes"
   - Flag specific instances with approximate timestamps
   - These are OBSERVABLE without CV precision

3. SHOULDER SHRUG MISMATCH:
   - Single-shoulder shrug during definitive statements (congruence indicator)
   - Full shoulder shrug with uncertain language (congruent)
   - Single-shoulder = possible lack of conviction
   - OBSERVABLE without CV precision

4. FACIAL-VERBAL MISMATCH:
   - Smiling while discussing negative topics (may be ironic - see culture filter)
   - Flat affect while expressing enthusiasm
   - Contempt flash during positive statements
   - OBSERVABLE but interpret with cultural context

5. TIMING ANALYSIS (LOW PRECISION without CV):
   - Delayed emotional reactions (estimate only - >0.5 sec after trigger)
   - Premature emotional displays (before trigger completes)
   - Sustained expressions (genuine emotions fade within 4-5 seconds)

Provide:
- SPECIFIC MISMATCH INSTANCES: [timestamp] - [verbal] vs [gestural]
- TIMING ASYNCHRONY EVENTS: [timestamp] - [description]
- DECEPTION PROBABILITY per mismatch: Low/Medium/High
- OVERALL GESTURAL CONGRUENCE: CONGRUENT / MIXED / INCONGRUENT
- TOP 3 MOST SIGNIFICANT MISMATCHES

Focus ONLY on gestural mismatch detection - be precise with timestamps."""


VISUAL_STRESS_CLUSTER_PROMPT = """Identify stress indicator clusters using NCI methodology.

CHASE HUGHES STRESS CLUSTER ANALYSIS:
Individual behaviors mean little - CLUSTERS of stress indicators are significant.
Context is crucial: always consider situational factors.

Identify and score stress indicator clusters:

1. PHYSICAL STRESS INDICATORS:
   - Increased blink rate
   - Pupil dilation (if visible)
   - Lip compression
   - Jaw tension
   - Throat clearing/swallowing
   - Shallow/rapid breathing (visible chest movement)

2. SELF-SOOTHING BEHAVIORS:
   - Face touching (nose, lips, ears)
   - Hair grooming
   - Neck touching/rubbing
   - Object manipulation (pen, jewelry)
   - Clothing adjustment

3. BARRIER BEHAVIORS:
   - Arm crossing
   - Object placement between self and camera
   - Leaning away
   - Foot withdrawal
   - Elbow closure

4. CONCEALMENT MARKERS:
   - Covering mouth while speaking
   - Looking down during statements
   - Turning away during key points
   - Hand hiding

5. CLUSTER SCORING:
   - 1-2 indicators: Background noise (normal)
   - 3-4 indicators: Attention warranted
   - 5+ indicators: Significant stress cluster - FLAG

For each identified cluster:
- TIMESTAMP: When did cluster appear?
- TRIGGER: What topic/question preceded it?
- INDICATORS: List all indicators present
- CLUSTER SCORE: Count of simultaneous indicators
- CONTEXT: Alternative explanations (cold room, nervousness, etc.)

Provide:
- Total stress clusters identified: [count]
- Highest cluster score: [count] at [timestamp]
- Topic correlation: Which topics triggered clusters?
- Overall stress assessment: LOW / MODERATE / HIGH / EXTREME

Focus ONLY on stress cluster identification."""


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

MICRO-TRIGGER WORD->GESTURE MAP:
Create explicit associations between trigger words and immediate physical responses:
Format: "[WORD/PHRASE]" -> [GESTURE] @ [TIMESTAMP]

Examples:
- "investment" -> collar touch @ 2:34
- "my wife" -> gaze aversion + blink cluster @ 4:12  
- "that night" -> postural shift backward @ 7:45

List at least 5-10 trigger word->gesture associations.
These are investigative gold - specific words that produced immediate stress responses.

**GESTURAL LATENCY ANALYSIS (Critical):**

Measure the TIME DELAY between verbal statements and accompanying gestures:

LATENCY CLASSIFICATION:
- NATURAL (0-200ms): Gesture precedes or accompanies word - AUTHENTIC
- DELAYED (200-500ms): Gesture follows word slightly - POSSIBLE REHEARSAL  
- FABRICATED (500ms+): Gesture notably trails speech - LIKELY DECEPTIVE

For each major claim or emphatic statement, document:
1. TIMESTAMP of verbal statement
2. TIMESTAMP of accompanying gesture  
3. LATENCY in milliseconds (estimate)
4. CLASSIFICATION: Natural/Delayed/Fabricated
5. What claim was being made?

LATENCY LOG FORMAT:
[MM:SS] "[statement]" -> [gesture type] @ [+Xms delay] = [classification]

Example entries:
- 1:23 "I absolutely did not take that money" -> emphatic hand chop @ +650ms = FABRICATED
- 2:45 "It was my idea" -> pointing gesture @ +100ms = NATURAL
- 3:12 "We had no knowledge" -> head shake @ +420ms = DELAYED

BASELINE LATENCY: Measure latency on non-sensitive topics first to establish subject's natural timing.

LATENCY SUMMARY:
- Topics with highest latency (most likely fabricated): [list]
- Average latency on sensitive vs non-sensitive topics: [comparison]
- Deception probability based on latency patterns: [Low/Medium/High]

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
# NCI/CHASE HUGHES MULTIMODAL SUB-PROMPTS (Stage 4 - run in parallel)
# Based on The Behavior Ops Manual and Six-Minute X-Ray methodologies
# =============================================================================

MULTIMODAL_FIVE_CS_PROMPT = """Apply Chase Hughes' Five C's Framework to behavioral analysis.

THE FIVE C'S FRAMEWORK (NCI Methodology):
The most important framework for accurate behavioral interpretation.

1. CHANGE (Most Important):
   - What is the subject's BASELINE behavior?
   - Where do their eyes go when accessing memory (their "home")?
   - What is their normal blink rate, posture, voice tone?
   - IDENTIFY DEVIATIONS from this baseline
   - Changes matter more than any checklist of behaviors

   Document:
   - Baseline established: [description]
   - Notable changes observed: [list with timestamps]
   - Change triggers: [what preceded each change]

2. CONTEXT:
   - What situational factors could explain behaviors?
   - Is the environment affecting behavior? (cold room = crossed arms)
   - Is the subject tired, hungry, ill?
   - Are there cultural factors to consider?
   - Is this a high-stakes situation naturally inducing stress?

   Document:
   - Contextual factors present: [list]
   - Behaviors potentially explained by context: [list]

3. CLUSTERS:
   - Single behaviors mean NOTHING - look for CLUSTERS
   - 3+ simultaneous indicators = meaningful cluster
   - Cluster + topic correlation = investigative focus point

   Document:
   - Clusters identified: [count]
   - Cluster compositions: [list indicators in each]
   - Cluster timing: [when did clusters appear]

4. CULTURE:
   - Cultural norms affect behavioral interpretation
   - Eye contact norms vary by culture
   - Personal space expectations differ
   - Gestures have different meanings across cultures

   Document:
   - Subject's apparent cultural background: [observation]
   - Behaviors that may be culturally normative: [list]
   - Potential misinterpretation risks: [list]

5. CHECKLIST (Use Last):
   - Only AFTER applying the first four C's
   - Compare observed clusters against known indicators
   - Weight findings by change significance and context

   Final assessment using all five C's:
   - Deception likelihood: LOW / MODERATE / HIGH
   - Confidence in assessment: LOW / MODERATE / HIGH
   - Key supporting evidence: [summary]

Provide a comprehensive Five C's analysis with clear documentation for each C."""


MULTIMODAL_BASELINE_DEVIATION_PROMPT = """Establish behavioral baseline and identify significant deviations.

CHASE HUGHES BASELINE METHODOLOGY:
"Detecting change is more important than any checklist."

1. ESTABLISH BASELINE (First 30-60 seconds or neutral topics):

   VISUAL BASELINE:
   - Eye "home" position: Where do eyes rest when thinking?
   - Blink rate baseline: Estimated BPM during calm moments
   - Resting facial expression: Neutral affect characteristics
   - Posture baseline: How do they naturally sit/stand?
   - Hand position baseline: Where do hands rest normally?
   - Movement baseline: Stillness vs. natural fidgeting level

   VOCAL BASELINE:
   - Speaking pace: Words per minute estimate
   - Pitch range: Normal vocal range
   - Volume level: Typical projection
   - Pause patterns: Natural hesitation frequency

   BEHAVIORAL BASELINE:
   - Engagement level: How do they show interest?
   - Response latency: How quickly do they typically respond?
   - Detail level: How much detail in normal answers?

2. TRACK DEVIATIONS:
   For each significant deviation from baseline, document:
   - TIMESTAMP: When did deviation occur?
   - BASELINE: What was the established norm?
   - DEVIATION: What changed?
   - TRIGGER: What topic/question preceded change?
   - MAGNITUDE: How significant was the deviation? (1-10)
   - RETURN: How quickly did subject return to baseline?

3. DEVIATION PATTERNS:
   - Which topics consistently trigger deviations?
   - Are deviations clustered or isolated?
   - Do deviations compound (one triggers another)?

4. RECOVERY ANALYSIS:
   - Fast recovery (<2 sec): Subject regained composure quickly
   - Slow recovery (>5 sec): Sustained stress response
   - No recovery: Persistent altered state

Provide:
- Baseline profile summary
- Top 5 most significant deviations with full documentation
- Deviation pattern analysis
- Topics requiring investigative follow-up
- Confidence in baseline establishment: LOW / MODERATE / HIGH"""


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

AUDIO_CREDIBILITY_PROMPT = """Assess vocal CREDIBILITY indicators with COGNITIVE LOAD ANALYSIS focus.

NOTE: This is a CREDIBILITY analysis, not "deception detection." Cognitive load indicates
mental effort - which may come from: complex recall, emotional difficulty, performance pressure,
second-language processing, or yes, fabrication. INTERPRET IN CONTEXT.

Analyze:
1. VOCAL STRESS: Pitch changes, tension, shakiness, throat clearing
2. SPEECH DISRUPTIONS: Hesitations, corrections, false starts
3. QUALIFIERS: "To be honest," "Frankly," "I swear" - distancing language
4. TENSE SHIFTS: Past/present inconsistencies
5. DETAIL PATTERNS: Over-elaborate vs sparse in specific sections

**COGNITIVE LOAD INDICATORS (Critical):**
6. SYNTAX BREAKDOWN POINTS:
   - Where does sentence structure collapse?
   - Where do complex sentences become fragmented?
   - Where does grammar fail under mental strain?

7. SPECIFICITY-TO-PHILOSOPHY PIVOTS:
   - Track when subject shifts from specific numbers/facts to general philosophy
   - "We made $2.3 million" -> "But what really matters is the vision"
   - Note exact timestamps of these pivots - they mark cognitive stress points

8. VERBAL PROCESSING OVERLOAD:
   - Unusual word choices (wrong register, malapropisms)
   - Sentence repairs mid-thought
   - Loss of pronoun consistency within same sentence
   - Abandoning complex answers for simple deflections

9. RESPONSE LATENCY PATTERNS:
   - Which topics produce immediate fluent answers?
   - Which topics produce delayed, fragmented responses?
   - Compare baseline response time to topic-specific delays

10. BASELINE-TO-STIMULUS DELTA SCORING (0-10):
    For each topic, score:
    - VOCAL BASELINE (0-10): Normal speech fluency/confidence
    - TOPIC REACTION (0-10): Stress level during specific topic
    - DELTA: [Reaction - Baseline]

    Example: "Topic: Partnership details. Baseline: 2 -> Reaction: 7 = Delta +5 (HIGH)"

For each indicator, provide:
- Timestamp reference
- Specific observation
- Cognitive Load Level: Low/Moderate/High/Overloaded
- DELTA SCORE for this topic
- Credibility concern: None/Low/Medium/High
- ALTERNATIVE EXPLANATIONS (emotional topic, complex recall, etc.)

CRITICAL: Identify the 3 MOMENTS OF HIGHEST COGNITIVE LOAD - these warrant investigation.

Describe areas of high/low credibility and POSSIBLE EXPLANATIONS (don't assume deception)."""


# =============================================================================
# NCI/CHASE HUGHES AUDIO SUB-PROMPTS (Stage 5 - run in parallel)
# Based on The Behavior Ops Manual and Six-Minute X-Ray methodologies
# =============================================================================

AUDIO_DETAIL_MOUNTAIN_VALLEY_PROMPT = """Analyze speech for "Detail Mountain and Valley" deception pattern.

CHASE HUGHES DETAIL MOUNTAIN/VALLEY THEORY:
Deceptive individuals often provide excessive detail about IRRELEVANT matters
while giving sparse detail about the CRITICAL event they're hiding.

Visualize their narrative as a topographical map:
- MOUNTAINS: Areas of excessive, unnecessary detail
- VALLEYS: Areas of suspiciously sparse detail

1. IDENTIFY DETAIL MOUNTAINS:
   - Where does subject provide excessive, irrelevant detail?
   - Examples: Describing car interior stitching, weather conditions, what they ate
   - These serve to: appear credible, stall, distract, pad narrative

   Document each mountain:
   - TOPIC: What were they elaborating on?
   - RELEVANCE: How relevant was this to the main point?
   - DETAIL LEVEL: Excessive (describe specifics)
   - TIMESTAMP: When did this occur?

2. IDENTIFY DETAIL VALLEYS:
   - Where does detail suddenly DROP?
   - The valley is usually WHERE THE CRITICAL EVENT OCCURRED
   - Signs: Vague language, skipped time, "and then... anyway..."
   - Glossing over: "things happened," "it was whatever"

   Document each valley:
   - TOPIC: What should have had more detail?
   - MISSING: What specific details were omitted?
   - TRANSITION: How did they skip past this?
   - TIMESTAMP: When did this occur?

3. MOUNTAIN-VALLEY PATTERN ANALYSIS:
   - Map the detail topography across the narrative
   - Where are the steepest drops (mountain → valley)?
   - What topics create valleys?
   - Is there a consistent valley around specific events/people/times?

4. DECEPTION ASSESSMENT:
   - Strong indicator: High detail before AND after, low detail during critical event
   - The valley indicates WHERE to focus investigative questions
   - Mountains indicate WHAT they've rehearsed/prepared

Provide:
- Detail mountains identified: [count] with descriptions
- Detail valleys identified: [count] with descriptions
- Most significant mountain-valley transition: [description]
- Recommended follow-up questions for each valley
- Pattern confidence: LOW / MODERATE / HIGH"""


AUDIO_MINIMIZING_LANGUAGE_PROMPT = """Detect minimizing and softening language as deception indicators.

CHASE HUGHES MINIMIZING LANGUAGE ANALYSIS:
Deceptive individuals often unconsciously minimize the severity of their actions
through specific word choices. This "verbal leakage" reveals guilt.

1. SEVERITY MINIMIZERS:
   Track substitutions that reduce perceived severity:
   - "hurt" instead of "killed" or "murdered"
   - "took" instead of "stole"
   - "pushed" instead of "attacked" or "assaulted"
   - "touched" instead of more severe contact
   - "disagreement" instead of "fight"
   - "borrowed" instead of "took without permission"
   - "accident" for intentional acts

   Document each instance:
   - WORD USED: [minimizing word]
   - LIKELY REALITY: [what was probably minimized]
   - CONTEXT: [surrounding statement]
   - TIMESTAMP: [when said]

2. HEDGE WORDS AND QUALIFIERS:
   - "I think" - creating deniability
   - "I believe" - distancing from certainty
   - "sort of" / "kind of" - softening
   - "maybe" / "probably" - avoiding commitment
   - "not really" - partial denial
   - "I guess" - feigned uncertainty

3. DISTANCING LANGUAGE:
   - Reduced use of "I" when describing own actions
   - Passive voice: "mistakes were made" vs "I made mistakes"
   - Third-person references to self
   - "One might think..." instead of "I think..."
   - "The situation" instead of "what I did"

4. DENIAL INTENSIFIERS (paradoxically suspicious):
   - "I would NEVER..."
   - "I SWEAR..."
   - "To be COMPLETELY honest..."
   - "I'm telling you the TRUTH..."
   - Unsolicited denial strengthening

5. QUANTIFIER ANALYSIS:
   - "a little" / "a bit" - minimizing amount
   - "just" - reducing significance ("I just...")
   - "only" - limiting scope
   - "few" instead of specific numbers
   - Vague quantities for things that should be specific

Provide:
- Total minimizing instances: [count]
- Most frequent minimizing pattern: [type]
- Topics with concentrated minimizing: [list]
- Severity assessment: Subject is likely minimizing [what]
- Recommended clarifying questions for each minimized topic"""


AUDIO_LINGUISTIC_HARVESTING_PROMPT = """Perform linguistic harvesting for personality and influence vectors.

CHASE HUGHES LINGUISTIC HARVESTING TECHNIQUE:
Analyze word choices to reveal values, priorities, and psychological vulnerabilities.

1. PRONOUN ANALYSIS:
   - I/me/my dominant: Self-focused, potentially narcissistic or anxious
   - We/us/our dominant: Team-oriented, collective identity
   - They/them dominant: External focus, blaming, distancing
   - You dominant: Engaging, persuading, or deflecting

   Count and categorize pronoun usage patterns.

2. SENSORY PREFERENCE (VAK Model):
   - VISUAL: "I see what you mean," "looks like," "picture this," "clearly"
   - AUDITORY: "I hear you," "sounds good," "listen," "tell me"
   - KINESTHETIC: "I feel," "grasp," "get a handle on," "touch base"

   Identify dominant sensory preference for rapport-building.

3. VALUE WORDS:
   - What positive descriptors do they use repeatedly?
   - What negative descriptors do they use?
   - What do they praise vs. criticize?
   - What standards do they apply to others?

   Map their value hierarchy.

4. TEMPORAL LANGUAGE:
   - Past-focused: Dwelling on history, regret, nostalgia
   - Present-focused: Immediate, reactive
   - Future-focused: Planning, anticipating, goal-oriented

5. POWER LANGUAGE:
   - Certainty words: "always," "never," "definitely" - confidence/dominance
   - Possibility words: "might," "could," "maybe" - flexibility/submission
   - Obligation words: "should," "must," "have to" - rule-orientation

6. EMOTIONAL VOCABULARY:
   - Range of emotional words used
   - Dominant emotional themes
   - Absent emotions (what do they NOT express?)

Provide:
- Pronoun profile: [breakdown with percentages]
- Sensory preference: VISUAL / AUDITORY / KINESTHETIC
- Top 5 value words (positive)
- Top 5 value words (negative)
- Temporal orientation: PAST / PRESENT / FUTURE
- Psychological leverage points identified
- Recommended influence approach based on linguistic profile"""

AUDIO_LIWC_PROMPT = """Perform LIWC (Linguistic Inquiry and Word Count) quantitative analysis on the transcript.

You are analyzing speech patterns using the LIWC framework. Provide ESTIMATED PERCENTAGES for each category
based on word frequency analysis. While you cannot do exact counts, estimate proportions relative to total word count.

TRANSCRIPT TO ANALYZE:
{transcript}

Provide quantitative LIWC metrics in this EXACT format:

=== LIWC QUANTITATIVE ANALYSIS ===

WORD COUNT STATISTICS:
- Total Words (estimated): [number]
- Words Per Sentence (avg): [number]
- Six+ Letter Words: [X]%
- Dictionary Words: [X]%

PRONOUN ANALYSIS:
- First Person Singular (I, me, my, mine): [X]%
- First Person Plural (we, us, our, ours): [X]%
- Second Person (you, your, yours): [X]%
- Third Person (he, she, they, them, it): [X]%
- Impersonal Pronouns (it, that, this): [X]%
- TOTAL Pronouns: [X]%

PSYCHOLOGICAL PROCESSES:

Affective Processes: [X]%
  - Positive Emotion: [X]%
  - Negative Emotion: [X]%
    - Anxiety: [X]%
    - Anger: [X]%
    - Sadness: [X]%

Cognitive Processes: [X]%
  - Insight (think, know, consider): [X]%
  - Causation (because, effect, hence): [X]%
  - Discrepancy (should, would, could): [X]%
  - Tentative (maybe, perhaps, guess): [X]%
  - Certainty (always, never, definitely): [X]%
  - Differentiation (but, else, except): [X]%

Social Processes: [X]%
  - Family: [X]%
  - Friends: [X]%
  - Social References: [X]%

PERSONAL CONCERNS:
- Work: [X]%
- Achievement: [X]%
- Leisure: [X]%
- Money: [X]%
- Religion: [X]%
- Death: [X]%

LINGUISTIC DIMENSIONS:
- Analytical Thinking (formal, logical): [score 0-100]
- Clout (confidence, leadership): [score 0-100]
- Authenticity (honest, personal): [score 0-100]
- Emotional Tone (positive vs negative): [score 0-100, 50=neutral]

TIME ORIENTATION:
- Past Focus: [X]%
- Present Focus: [X]%
- Future Focus: [X]%

DRIVES:
- Affiliation (social bonding): [X]%
- Achievement (success, winning): [X]%
- Power (dominance, control): [X]%
- Reward (positive goals): [X]%
- Risk (danger, threat): [X]%

DECEPTION INDICATORS (LIWC-based):
- Self-Reference Ratio: [High/Normal/Low] - Low self-reference may indicate distancing from lies
- Negative Emotion Density: [High/Normal/Low] - Liars show increased negative emotion
- Cognitive Complexity: [High/Normal/Low] - Truthful accounts have higher complexity
- Exclusive Words (but, except, without): [High/Normal/Low] - Low exclusive words suggest rehearsed narrative

=== PRONOUN DRIFT TRACKER (Critical Deception Indicator) ===

Track PRONOUN SHIFTS across the transcript timeline:
- When does subject shift from "I" to "we"? (diffusing personal responsibility)
- When does subject shift from "we" to "they"? (distancing from group)
- When does subject use passive voice? (removing agency)
- When does subject switch from naming people to "someone" or "people"?

DRIFT PATTERN LOG:
For each significant pronoun shift, document:
1. TIMESTAMP/CONTEXT: When in the narrative did this occur?
2. BEFORE: What pronoun was being used?
3. AFTER: What did it shift to?
4. TOPIC: What subject was being discussed?
5. DECEPTION PROBABILITY: Based on topic sensitivity

Example shifts to flag:
- "I made the decision" -> "The decision was made" (passive distancing)
- "I met with John" -> "We met with some people" (specificity reduction)
- "My team did X" -> "They did X" (disowning responsibility)
- "I told her" -> "It was communicated" (bureaucratic distancing)

PRONOUN DRIFT SUMMARY:
- Total significant shifts detected: [number]
- Topics with highest drift frequency: [list]
- Deception probability assessment: [Low/Medium/High]

=== KEY FINDINGS ===
List 3-5 psychologically significant patterns from the metrics above.

=== INVESTIGATIVE RELEVANCE ===
How do these linguistic patterns inform the behavioral profile?

Be systematic. Provide actual percentage estimates based on your analysis of word frequencies."""



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

**STEP 1: IDENTIFY SUBJECT TYPE** (from Subject Identification analysis)
- Physical Threat Context (law enforcement, violence history)
- Financial/Influence Context (trader, influencer, executive, crypto)
- Political/Activism Context (politician, organizer, lobbyist)
- General/Unknown Context

**STEP 2: RATE METRICS 0-100 with evidence:**
- Volatility Risk: [score] | [evidence]
- Manipulation Capacity: [score] | [evidence]
- Compliance Likelihood: [score] | [evidence]
- Stress Resilience: [score] | [evidence]
- Ethical Boundaries: [score] | [evidence]

**STEP 3: CALIBRATED OVERALL THREAT LEVEL**
Apply context-specific weighting:
- Physical Context: Weight Volatility Risk 2x
- Financial/Influence Context: Weight Manipulation Capacity 2x, Ethical Boundaries 1.5x
- Political Context: Weight Manipulation Capacity 1.5x, Ethical Boundaries 2x
- General: Equal weighting

THREAT LEVEL: LOW / MODERATE / HIGH / CRITICAL
(Explicitly state which weighting was applied and why)

Justify with:
- Risk of violence
- Risk of flight
- Risk of evidence destruction
- Cooperation likelihood
- Deception capability
- CONTEXT-SPECIFIC RISKS (market manipulation, influence operations, etc.)

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

SYNTHESIS_CONTRADICTIONS_PROMPT = """Analyze contradictions between modalities and personality traits (Hot Spots).

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

PART A - MODALITY CONFLICTS:
Identify conflicts between:
- Visual vs Audio (confident voice but anxious eyes)
- Verbal content vs Non-verbal behavior
- Beginning vs End behavioral drift
- Self-presentation vs Micro-expression leakage

PART B - TRAIT CONTRADICTIONS:
Identify and reconcile seemingly contradictory personality assessments:
- High Openness vs Lone Wolf/Isolationist tendencies
- Extroverted behavior vs Social Anxiety markers
- Confident presentation vs Low Self-Esteem indicators
- Empathic claims vs Narcissistic traits
- Agreeable demeanor vs Manipulative patterns

For seemingly contradictory traits, explain:
1. APPARENT CONTRADICTION: [Trait A] vs [Trait B]
2. RECONCILIATION: How both can coexist (e.g., "High Openness to IDEAS but low Openness to PEOPLE explains intellectual curiosity combined with social isolation")
3. FUNCTIONAL EXPLANATION: What adaptive purpose does this combination serve?

For EACH contradiction found:
1. CONFLICTING SIGNALS: [specific observations]
2. UNIFIED INTERPRETATION: Why do both exist simultaneously?
3. PSYCHOLOGICAL INSIGHT: What this reveals about strategy/psychology

Example resolution:
BAD: "Subject shows both submissive and dominant behaviors."
GOOD: "Subject employs Social Engineering - using vocal submission to appear non-threatening while maintaining physical dominance to control the interaction frame."

BAD: "High Openness but also Lone Wolf tendencies."
GOOD: "Subject demonstrates high Openness to EXPERIENCES and IDEAS (intellectual curiosity, philosophical discussion) while simultaneously showing low Agreeableness and selective social engagement. This combination suggests an intellectual who values depth over breadth in relationships - open-minded about concepts but choosy about people."

Contradictions ARE the profile. Resolve them, don't just list them."""

SYNTHESIS_RED_TEAM_PROMPT = """Perform Red Team analysis (self-critique) and Devil's Advocate check.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

PART A - DEVIL'S ADVOCATE (Confirmation Bias Check):
Challenge every major conclusion. For each key finding in the profile:

1. CONTRADICTING EVIDENCE:
   - What observed behaviors CONTRADICT our main conclusions?
   - List specific evidence that argues AGAINST the profile

2. ALTERNATIVE PROFILE:
   - If we're completely wrong, what personality would these behaviors suggest?
   - Present the most plausible alternative interpretation

3. DISCONFIRMATION TEST:
   - What evidence would DISPROVE our current assessment?
   - What would we expect to see if our profile is incorrect?

PART B - STANDARD RED TEAM ANALYSIS:
Identify THREE reasons why this analysis might be WRONG:

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

CONFIDENCE CALIBRATION:
- Areas of HIGH confidence: [list]
- Areas of LOW confidence (hedging recommended): [list]
- Conclusions that should be treated as hypotheses only: [list]

This ensures investigators don't over-rely on the profile."""


SYNTHESIS_IRONY_MEME_FILTER_PROMPT = """Apply Internet Culture / Irony Filter for digital-native subjects.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

PURPOSE: Many subjects - especially crypto traders, tech workers, streamers, influencers,
and those active in online communities - use IRONIC, MEMETIC, or IN-GROUP language that
can be misinterpreted as genuine distress, regret, or shame when it is actually:
- Performative self-deprecation (common in tech/crypto culture)
- In-group signaling / community bonding
- Ironic hyperbole for entertainment
- Meme references (may sound distressed but are cultural references)

CRITICAL QUESTION: Is this subject INTERNET-NATIVE?

1. DIGITAL CULTURE INDICATORS:
   - Uses crypto/tech/gaming jargon ("doomer," "ngmi," "cope," "based," "rugged")
   - References memes or internet culture
   - Self-deprecating humor about losses/failures (common in trading culture)
   - Hyperbolic language ("absolutely destroyed," "life ruined," said with smile)
   - Uses community catchphrases

2. STATEMENTS FLAGGED AS "DISTRESS" - RE-EVALUATE:
   Review any statements flagged as indicating shame, regret, or distress.
   For each, assess:
   - WAS IT SAID IRONICALLY? (vocal tone, accompanying smile, exaggeration)
   - IS IT A MEME REFERENCE? (recognize "doomer," "cope," "diamond hands," etc.)
   - IS IT IN-GROUP PERFORMANCE? (bonding with audience through shared experience)
   - IS IT GENUINE? (if so, what supporting evidence?)

3. RECALIBRATION TABLE:
   For each potentially ironic statement:
   | Statement | Original Flag | Irony Likelihood | Revised Interpretation |
   |-----------|---------------|------------------|------------------------|
   | Example   | Shame/Regret  | HIGH (smiling)   | Entertainment/bonding  |

4. CULTURAL CONTEXT ADJUSTMENTS:
   If subject is identified as crypto/trading community:
   - Bearish language ("doomer," "ngmi") = In-group mood expression, NOT depression
   - Chart commentary ("disgusting," "beautiful") = Standard vernacular, NOT literal emotion
   - Loss terminology ("rugged," "rekt") = Neutral technical terms
   - Self-deprecating loss stories = Entertainment, often exaggerated

   If subject is tech/startup community:
   - Crisis language ("everything is on fire," "dumpster fire") = Normal vernacular
   - Self-deprecation ("imposter syndrome," "I have no idea what I'm doing") = Often performative

   If subject is gaming/streaming community:
   - Rage expressions = Often theatrical for entertainment
   - Self-roasts = Community bonding ritual

5. DECEPTION ANALYSIS RECALIBRATION:
   IMPORTANT: If subject shows incongruence (e.g., smiling while discussing losses),
   in internet-native subjects this is often:
   - NOT duping delight
   - NOT hidden satisfaction
   - IS: Ironic self-aware performance for audience entertainment

   Re-evaluate any "incongruence" flags for irony/performance.

6. FINAL FILTER OUTPUT:
   - Subject digital nativity: LOW / MEDIUM / HIGH
   - Statements requiring irony recalibration: [list]
   - Revised deception/distress assessments: [list changes]
   - Cultural blind spots in original analysis: [list]

This filter prevents false positives in populations where irony and meme culture are normative."""


# =============================================================================
# NCI/CHASE HUGHES SYNTHESIS SUB-PROMPTS (Stage 6)
# Based on The Behavior Ops Manual and Six-Minute X-Ray methodologies
# =============================================================================

SYNTHESIS_FATE_MODEL_PROMPT = """Apply the FATE Model to understand subject's core motivational drivers.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

CHASE HUGHES FATE MODEL:
The FATE Model identifies the four primal psychological drivers that govern human behavior.
Understanding which driver is PRIMARY for this subject enables targeted influence.

Analyze all previous data to determine the subject's FATE profile:

1. FOCUS - What captures their attention?
   - What topics make them lean in, engage more?
   - What do they return to repeatedly?
   - What do they monitor or track obsessively?
   - Where do their eyes go when not directly engaged?

   Evidence from analyses:
   - [cite specific observations]
   Focus Driver Strength: LOW / MODERATE / HIGH / PRIMARY

2. AUTHORITY - How do they relate to power?
   - Do they seek to establish dominance?
   - Do they defer to authority figures?
   - Do they challenge or respect hierarchies?
   - How do they respond to being questioned?
   - Do they use power language ("you should," "you must")?

   Evidence from analyses:
   - [cite specific observations]
   Authority Driver Strength: LOW / MODERATE / HIGH / PRIMARY

3. TRIBE - What group affiliations drive them?
   - Who do they identify with? ("we" language)
   - What groups do they exclude? ("they" language)
   - Do they seek belonging or independence?
   - How important is social approval to them?
   - What loyalty signals do they display?

   Evidence from analyses:
   - [cite specific observations]
   Tribe Driver Strength: LOW / MODERATE / HIGH / PRIMARY

4. EMOTION - What emotional triggers are active?
   - What topics produce emotional reactions?
   - What fears are evident?
   - What desires are expressed?
   - What emotional needs are unmet?
   - What emotional manipulation tactics do they use?

   Evidence from analyses:
   - [cite specific observations]
   Emotion Driver Strength: LOW / MODERATE / HIGH / PRIMARY

FATE PROFILE SUMMARY:
- PRIMARY DRIVER: [F/A/T/E] - [description]
- SECONDARY DRIVER: [F/A/T/E] - [description]
- VULNERABILITY VECTOR: How to leverage primary driver
- RESISTANCE VECTOR: What approach would backfire

INFLUENCE RECOMMENDATIONS:
Based on FATE profile, provide specific tactics:
- Opening approach
- Rapport-building strategy
- Compliance-gaining technique
- Potential resistance and countermeasures

Focus on synthesizing FATE insights from all available data."""


SYNTHESIS_NCI_DECEPTION_SUMMARY_PROMPT = """Create comprehensive NCI-methodology deception assessment summary.

ANALYSIS DATA TO SYNTHESIZE:
{previous_analyses}

Synthesize ALL NCI/Chase Hughes deception indicators into a unified assessment.

1. BEHAVIORAL TABLE OF ELEMENTS (BTE) SUMMARY:
   - Total BTE score from visual analysis: [X]
   - Threshold assessment: Below 8 / 8-12 / 12+
   - Primary indicators contributing to score

2. BLINK RATE ASSESSMENT:
   - Baseline established: [X] BPM
   - Peak deviation: [X] BPM
   - Topics correlated with elevated blinking

3. FIVE C'S INTEGRATION:
   - CHANGE: Most significant baseline deviations
   - CONTEXT: Factors that may explain behaviors
   - CLUSTERS: Stress clusters identified
   - CULTURE: Cultural considerations
   - CHECKLIST: Final indicator assessment

4. DETAIL MOUNTAIN/VALLEY MAP:
   - Mountains (excessive detail): [topics]
   - Valleys (sparse detail): [topics]
   - Investigative focus areas

5. GESTURAL MISMATCH SUMMARY:
   - Timing asynchrony instances: [count]
   - Verbal-gestural contradictions: [count]
   - Most significant mismatches

6. LINGUISTIC INDICATORS:
   - Minimizing language instances: [count]
   - Pronoun pattern anomalies
   - Distancing language presence

7. STRESS CLUSTER CORRELATION:
   - Topics triggering multiple stress indicators
   - Cross-modal stress alignment

8. DECEPTION CONFLICT MATRIX:
   Explicitly compare deception signals across modalities:

   | Modality          | Deception Signal | Confidence | Key Evidence |
   |-------------------|------------------|------------|--------------|
   | Visual (BTE)      | LOW/MED/HIGH     | %          | [evidence]   |
   | Blink Rate        | LOW/MED/HIGH     | %          | [evidence]   |
   | Gestural Mismatch | LOW/MED/HIGH     | %          | [evidence]   |
   | Vocal Indicators  | LOW/MED/HIGH     | %          | [evidence]   |
   | Linguistic (LIWC) | LOW/MED/HIGH     | %          | [evidence]   |
   | Five C's          | LOW/MED/HIGH     | %          | [evidence]   |
   | Detail Mt/Valley  | LOW/MED/HIGH     | %          | [evidence]   |

   **MANDATORY CONFLICT RESOLUTION** (DO NOT SKIP THIS SECTION):
   When modalities disagree (e.g., BTE=HIGH but LIWC=LOW), you MUST:
   1. Explicitly state the conflict: "CONFLICT: BTE indicates HIGH deception (score 16) while LIWC indicates LOW (Authenticity 88, high self-reference)"
   2. Analyze WHY they disagree - what could cause this divergence?
   3. Apply hierarchy: Involuntary > Voluntary, Clusters > Isolated, Cross-modal > Single-modal
   4. Issue a RULING with specific reasoning

   CONFLICTS IDENTIFIED:
   - [Modality A] vs [Modality B]: [Describe conflict]
   - RULING: [Which modality is more reliable for this specific case] because [reasoning]
   - Alternative interpretation: [What might explain the discrepancy]

   AGREEMENT AREAS:
   - [Areas where multiple modalities converge on same conclusion]

   **BLINK RATE DATA RECONCILIATION:**
   If CV-detected blink rate differs from LLM-estimated rate:
   - CV data is GROUND TRUTH (measures actual eye closures via EAR algorithm)
   - Use CV values for all blink rate claims
   - Note if LLM hallucinated higher values (common failure mode)

INTEGRATED DECEPTION ASSESSMENT:

OVERALL DECEPTION PROBABILITY: LOW / MODERATE / HIGH / VERY HIGH
Confidence: LOW / MODERATE / HIGH

PRIMARY DECEPTION INDICATORS:
1. [Most significant indicator with evidence]
2. [Second most significant]
3. [Third most significant]

LIKELY TOPICS OF DECEPTION:
- [Topic 1]: Evidence summary
- [Topic 2]: Evidence summary

RECOMMENDED FOLLOW-UP QUESTIONS:
1. [Question targeting valley/sparse detail area]
2. [Question targeting stress cluster topic]
3. [Question targeting minimizing language topic]

ALTERNATIVE EXPLANATIONS:
- [Non-deceptive explanation for observed behaviors]
- Probability of alternative: [%]

This assessment follows NCI methodology: stress indicators suggest deception probability,
but NO behavior definitively proves deception. Use as investigative guidance only."""

SYNTHESIS_FINAL_INTEGRATION_PROMPT = """Generate an FBI-style CASE FILE based on all behavioral analyses.

ALL ANALYSIS DATA (includes subject identification):
{previous_analyses}

SYNTHESIS SUB-ANALYSES:
{synthesis_results}

**OUTPUT AS A STRUCTURED CASE FILE (DASHBOARD-FIRST FORMAT):**
Executive decisions at TOP. Raw data as APPENDICES at bottom.

═══════════════════════════════════════════════════════════════════════════════
                        BEHAVIORAL ANALYSIS UNIT
                           CASE FILE REPORT
═══════════════════════════════════════════════════════════════════════════════

CLASSIFICATION: UNCLASSIFIED // FOR EDUCATIONAL USE ONLY
ANALYST: Automated Behavioral Analysis System

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      SECTION 1: EXECUTIVE DASHBOARD                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

**ONE-LINE VERDICT:**
[Single sentence: "Subject displays [HIGH/MODERATE/LOW] deception indicators regarding [topic]"]

**THREAT MATRIX:**
┌────────────────────┬──────────┬─────────────────────────────────┐
│ Category           │ Level    │ Key Evidence                    │
├────────────────────┼──────────┼─────────────────────────────────┤
│ Overall Threat     │ [LEVEL]  │ [One-liner summary]             │
│ Physical Threat    │ [Level]  │ [Brief evidence]                │
│ Financial/Fraud    │ [Level]  │ [Brief evidence]                │
│ Manipulation Risk  │ [Level]  │ [Brief evidence]                │
│ Credibility        │ [Level]  │ [Brief evidence]                │
└────────────────────┴──────────┴─────────────────────────────────┘

**KEY ANOMALIES (Top 3 - Investigate These First):**
1. ⚠️ [Timestamp] - [Anomaly description] - [Why critical]
2. ⚠️ [Timestamp] - [Anomaly description] - [Why critical]
3. ⚠️ [Timestamp] - [Anomaly description] - [Why critical]

**CV-VALIDATED STRESS TRIGGERS:**
[From Trigger-Response Map if available - exact words that caused blink spikes]
- [Timestamp]: "[exact word/phrase]" → [X]x baseline blink spike
- [Timestamp]: "[exact word/phrase]" → [X]x baseline blink spike

**EXECUTIVE SUMMARY:**
[2-3 sentence behavioral overview]

───────────────────────────────────────────────────────────────────────────────
SECTION 2: SUBJECT IDENTIFICATION
───────────────────────────────────────────────────────────────────────────────

- NAME: [If identified] or "UNIDENTIFIED SUBJECT"
- ALIASES: [Online handles, nicknames]
- DESCRIPTION: [Age, ethnicity, distinguishing features]
- OCCUPATION: [Role/profession if identifiable]
- DOMAIN: [Crypto/Finance/Tech/Gaming/Military/Academic/Other]
- ID CONFIDENCE: [HIGH/MEDIUM/LOW/UNIDENTIFIED]

───────────────────────────────────────────────────────────────────────────────
SECTION 3: CREDIBILITY ANALYSIS (The "Why")
───────────────────────────────────────────────────────────────────────────────

**OVERALL VERACITY:** [TRUTHFUL / MIXED / DECEPTIVE]
**BTE SCORE:** [Score] ([Below 8: Low | 8-12: Moderate | 12+: High concern])
**CONFIDENCE:** [HIGH/MEDIUM/LOW]

**CRITICAL DECEPTION INDICATORS:**
| # | Timestamp | Indicator | Evidence | Weight |
|---|-----------|-----------|----------|--------|
| 1 | [Time]    | [Type]    | [Detail] | HIGH   |
| 2 | [Time]    | [Type]    | [Detail] | HIGH   |
| 3 | [Time]    | [Type]    | [Detail] | MEDIUM |

**BODY-VERBAL CONFLICTS:**
| Timestamp | Said | Body Did | Interpretation |
|-----------|------|----------|----------------|
| [Time]    | "[quote]" | [behavior] | [meaning] |

**COGNITIVE LOAD HOTSPOTS:**
Topics that triggered measurable stress response:
- [Topic 1]: [Evidence]
- [Topic 2]: [Evidence]

───────────────────────────────────────────────────────────────────────────────
SECTION 4: CONFLICT RESOLUTION (Modality Disagreements)
───────────────────────────────────────────────────────────────────────────────

**EXPLICIT CONFLICTS RESOLVED:**

| Conflict | Analysis A | Analysis B | Ruling | Reasoning |
|----------|------------|------------|--------|-----------|
| Blink Rate | LLM: [X] BPM | CV: [Y] BPM | CV (ground truth) | [why] |
| Deception | BTE: HIGH | LIWC: LOW | [ruling] | [stress ≠ deception?] |
| Neuroticism | Facial Etching: LOW | Behavior: HIGH | Dynamic (behavior) | [why] |

**LEAD INVESTIGATOR RULING:**
[Final credibility determination with evidence-based reasoning]

───────────────────────────────────────────────────────────────────────────────
SECTION 5: OPERATIONAL RECOMMENDATIONS
───────────────────────────────────────────────────────────────────────────────

**INTERVIEW APPROACH:** [Strategy with rationale]

**PSYCHOLOGICAL LEVERAGE POINTS:**
1. [Point with rationale]
2. [Point with rationale]
3. [Point with rationale]

**PRESSURE RESPONSE PREDICTION:** [How subject reacts when challenged]

**DE-ESCALATION TRIGGERS:** [What to avoid]

**INVESTIGATIVE PRIORITIES:**
1. [Primary verification need with rationale]
2. [Secondary focus]
3. [Tertiary focus]

**STATEMENTS REQUIRING VERIFICATION:**
- [Timestamp]: "[Specific claim]"
- [Timestamp]: "[Specific claim]"

═══════════════════════════════════════════════════════════════════════════════
                              APPENDICES
                     (Raw Data for Forensic Review)
═══════════════════════════════════════════════════════════════════════════════

───────────────────────────────────────────────────────────────────────────────
APPENDIX A: CHRONOLOGICAL HOT SPOT MAP
───────────────────────────────────────────────────────────────────────────────

| Time | Event Type | Signals | Interpretation |
|------|------------|---------|----------------|
| [Time] | [Type] | [Signals] | [Meaning] |
[Continue for all significant moments, sorted by timestamp]

───────────────────────────────────────────────────────────────────────────────
APPENDIX B: PSYCHOLOGICAL PROFILE
───────────────────────────────────────────────────────────────────────────────

PRIMARY ARCHETYPE: [Type]
SECONDARY TRAITS: [Traits]

**Big Five:**
- Openness: [Score]/100 - [Evidence]
- Conscientiousness: [Score]/100 - [Evidence]
- Extraversion: [Score]/100 - [Evidence]
- Agreeableness: [Score]/100 - [Evidence]
- Neuroticism: [Score]/100 - [Evidence]

**Dark Triad:**
- Narcissism: [Score]/100 - [Evidence]
- Machiavellianism: [Score]/100 - [Evidence]
- Psychopathy: [Score]/100 - [Evidence]

MBTI HYPOTHESIS: [Type] ([Confidence])

───────────────────────────────────────────────────────────────────────────────
APPENDIX C: DOMAIN-SPECIFIC LEXICON ADJUSTMENTS
───────────────────────────────────────────────────────────────────────────────

Subject Domain: [Crypto/Finance/Tech/Gaming/Military/Academic]

Words recalibrated from literal to domain meaning:
| Word | Literal | Domain Meaning | Weight Adjustment |
|------|---------|----------------|-------------------|
| [word] | [literal] | [domain] | [neutral/reduced] |

═══════════════════════════════════════════════════════════════════════════════
                           END OF CASE FILE
═══════════════════════════════════════════════════════════════════════════════

**CRITICAL WEIGHTING (Apply throughout):**
- Body language signals carry 5x weight of verbal content
- Micro-expressions carry 3x weight of macro-expressions
- Involuntary behaviors outweigh voluntary presentations
- Cross-modal conflicts are primary deception indicators
- When verbal contradicts non-verbal, TRUST THE BODY
- Facial Etching analysis is SUPPLEMENTARY only (low scientific validity for personality prediction)
  Use it for physical baseline description, NOT as primary personality evidence

**CONFLICT RESOLUTION PROTOCOL (MANDATORY - DO NOT SKIP):**
You are the Lead Investigator. When analyses conflict, you MUST RESOLVE them with explicit reasoning.

COMMON CONFLICTS TO ADDRESS:
- BTE Score HIGH (12+) vs LIWC/Five C's showing LOW deception: This often means the subject is experiencing STRESS (body) while being TRUTHFUL (language). Stress ≠ Deception.
- High blink rate claims vs CV data: ALWAYS use CV-detected blink rates as ground truth. LLMs frequently hallucinate 2-5x higher blink rates.

RESOLUTION HIERARCHY:
1. CV-measured data > LLM-estimated data (blink rates, etc.)
2. DYNAMIC behavioral data > STATIC physiological data (critical!)
   - Dynamic: Blinks, sweating, fidgeting, voice stress, micro-expressions
   - Static: Facial etching, wrinkle patterns, bone structure
   - If Facial Etching says "Low Neuroticism" but dynamic behavior shows "High stress" → TRUST DYNAMIC
   - Static features show long-term tendencies; dynamic features show current state
3. Involuntary behaviors > Voluntary behaviors
4. Cluster patterns > Isolated incidents
5. Cross-modal alignment > Single-modal signals
6. Linguistic authenticity markers (LIWC) can explain away visual stress as emotional recall, not deception

REQUIRED FORMAT for each conflict:
"CONFLICT: [Modality A] indicates [X] while [Modality B] indicates [Y]
RULING: [Your adjudication]
REASONING: [Specific evidence-based explanation]"

If BTE=HIGH but LIWC/Five C's=LOW, consider:
- Is subject recalling genuinely painful memories? (stress without deception)
- Are stress indicators clustered around specific topics? (may indicate emotional sensitivity, not lies)
- Does linguistic analysis show high authenticity/self-reference? (truth markers)

**QUANTITATIVE BACKING REQUIREMENT:**
All personality/behavioral claims MUST cite specific LIWC metrics:
- BAD: "Subject uses intellectualization"
- GOOD: "Subject uses intellectualization (LIWC Cognitive Process: 19.5%, Analytical Thinking: 42)"
Include at least 3 LIWC metrics in the Psychological Profile section.

**REDUNDANCY ELIMINATION (CRITICAL):**
The input analyses may describe the same behavioral observation multiple times.
- DO NOT repeat the same observation more than once in your output
- If "nose rub at 0:56" appears in 5 different analyses, mention it ONCE with "[5 analyses flagged]"
- Consolidate similar observations: "Multiple self-soothing gestures (neck rub, nose touch, eye blocking) clustered at 1:12"
- Focus on PATTERNS not individual instances
- Your output should be CONCISE - if an observation has been noted, do not describe it again

Generate the complete case file following this exact structure.
Reference the subject identification analysis to fill Section 1."""


# =============================================================================
# SUBJECT IDENTIFICATION PROMPT (For Case File)
# =============================================================================

SUBJECT_IDENTIFICATION_PROMPT = """Identify and describe the subject in this video for a case file.

Your task is to create a SUBJECT IDENTIFICATION section for an investigative case file.

**IDENTIFICATION ATTEMPT:**
1. Is this a KNOWN PUBLIC FIGURE? (celebrity, influencer, politician, business figure)
   - If YES: Provide their name, known aliases/handles, and why you believe this identification
   - If NO or UNCERTAIN: State "UNKNOWN SUBJECT" and proceed with physical description only

2. CONFIDENCE LEVEL for identification: HIGH / MEDIUM / LOW / UNIDENTIFIED

**PHYSICAL DESCRIPTION (Provide regardless of identification):**
- Apparent gender: [Male/Female/Other]
- Estimated age range: [e.g., 25-35]
- Ethnicity/racial presentation: [description]
- Hair: [color, style, length]
- Facial hair: [if applicable]
- Distinguishing features: [scars, tattoos, glasses, etc.]
- Build/physique: [if visible]

**PRESENTATION:**
- Attire: [what they're wearing]
- Grooming: [well-groomed, casual, disheveled, etc.]
- Accessories: [jewelry, headwear, etc.]
- Apparent socioeconomic indicators

**VOCAL IDENTIFICATION:**
- Accent: [regional, national origin if identifiable]
- Speech pattern: [formal, casual, technical jargon usage]
- Estimated education level based on vocabulary

**DIGITAL PRESENCE (if identifiable):**
- Known social media handles
- Known platforms/channels
- Estimated following/reach
- Known associates (if public figure)

**OUTPUT FORMAT:**
If identified, start with:
SUBJECT IDENTIFIED: [Full Name]
CONFIDENCE: [HIGH/MEDIUM/LOW]
BASIS FOR IDENTIFICATION: [how you recognized them]

If not identified:
SUBJECT: UNIDENTIFIED INDIVIDUAL
DESIGNATION: [SUBJ-001 or similar]

Then provide the physical description sections."""



# =============================================================================
# STAGE 0: SUBJECT ID & BASELINE (Run FIRST before other analyses)
# =============================================================================

BASELINE_ESTABLISHMENT_PROMPT = """Establish behavioral BASELINE using DYNAMIC BASELINE FINDING.

**CRITICAL: This analysis MUST run before anomaly detection.**
The baseline you establish here will be used by all subsequent analyses to identify DEVIATIONS.

**DYNAMIC BASELINE FINDING PROTOCOL:**
DO NOT automatically use the first 30 seconds. Instead:

1. SCAN THE ENTIRE VIDEO FIRST for these segments:
   - Lowest movement/fidgeting
   - Most neutral facial expression
   - Steadiest vocal tone (if audible)
   - Calmest topic being discussed

2. IDENTIFY THE BEST BASELINE SEGMENT:
   - May be at the START (introductions, small talk)
   - May be in the MIDDLE (discussing factual/neutral topics)
   - May be at the END (wrap-up, pleasantries)
   - AVOID: Segments where subject is clearly agitated, defensive, or animated

3. DOCUMENT YOUR BASELINE SELECTION:
   - Segment used: [Start time] - [End time]
   - Why selected: [Reasoning - "lowest fidget rate," "neutral topic," etc.]
   - Why NOT first 30 seconds (if applicable): [e.g., "Subject started with high-energy pitch"]

FOCUS ON YOUR SELECTED BASELINE SEGMENT:

1. VISUAL BASELINE:
   - EYE "HOME" POSITION: Where do eyes rest when subject is thinking/neutral?
   - BLINK RATE: Estimate comfortable baseline BPM
   - RESTING FACE: What is their neutral expression? Any permanent asymmetries?
   - POSTURE: Natural sitting/standing position
   - HAND POSITION: Where do hands naturally rest?
   - MOVEMENT LEVEL: Natural fidget frequency (some people are naturally fidgety)

2. VOCAL BASELINE:
   - SPEAKING PACE: Words per minute during neutral content
   - PITCH RANGE: Normal comfortable range
   - VOLUME: Default projection level
   - PAUSE FREQUENCY: Natural hesitation patterns

3. BEHAVIORAL BASELINE:
   - SELF-TOUCH FREQUENCY: How often do they touch face/hair at baseline?
   - GESTURE STYLE: Are they naturally expressive or reserved?
   - ENGAGEMENT INDICATORS: How do they show interest normally?

4. ENVIRONMENTAL FACTORS (Critical for false positive reduction):
   - HEADPHONES/HEADSET: Subject wearing headphones that may cause ear/head touching?
   - HAT/HEADWEAR: May cause head touching or adjustments
   - FACIAL HAIR: Beard/mustache that may be scratched habitually
   - GLASSES: May be adjusted frequently
   - VISIBLE DISCOMFORT: Signs of physical discomfort (cold room, uncomfortable chair)
   - LIGHTING: Bright lights that may cause squinting or eye rubbing

**OUTPUT FORMAT:**

=== BASELINE PROFILE ===
SEGMENT ANALYZED: [timestamp range]
BASELINE QUALITY: HIGH/MEDIUM/LOW (based on how neutral the segment appeared)

VISUAL BASELINE:
- Eye home position: [description]
- Baseline blink rate: ~[X] BPM
- Resting expression: [description]
- Posture baseline: [description]
- Hand position default: [description]
- Natural movement level: LOW/MEDIUM/HIGH

VOCAL BASELINE:
- Speaking pace: ~[X] WPM
- Pitch: [description]
- Volume: [description]
- Pause pattern: [description]

HABITUAL BEHAVIORS (NOT stress indicators):
- [List behaviors that appear to be habitual, not stress-induced]
- Example: "Frequent beard stroking - appears habitual, not tied to topic"

PHYSICAL DISCOMFORT FACTORS:
- [List any environmental/equipment factors that may cause movements]
- Example: "Wearing over-ear headphones - expect ear/head touching"

=== DEVIATION DETECTION GUIDANCE ===
For subsequent analyses, ONLY flag behaviors that DEVIATE from this baseline.
DO NOT score habitual behaviors as stress indicators."""


KINESIC_EVENT_LOG_PROMPT = """Generate a single KINESIC EVENT LOG (replaces all separate behavioral analyses).

**PURPOSE: ONE master chronological log of ALL behavioral events.**
This consolidates FACS, BTE, Body Language, Gestural Mismatch, and Gesture Timeline into
a single source of truth. Each behavior is coded ONCE with all metrics attached.

**BASELINE CONTEXT (if available):**
{baseline_context}

=== KINESIC EVENT LOG ===

For EVERY significant behavioral event, create ONE entry with ALL relevant data:

| Time | Event | FACS | BTE | Category | Verbal Sync | Touch Target | Significance |

COLUMN DEFINITIONS:
- **Time**: MM:SS timestamp
- **Event**: Brief description of behavior
- **FACS**: Action Unit codes (AU1, AU4, AU12, etc.) or "-" if not facial
- **BTE**: Score 0-3 (0=baseline, 1=mild, 2=clear, 3=pronounced)
- **Category**: PACIFIER / BARRIER / STRESS / EMPHASIS / CONCEALMENT / MISMATCH / EQUIPMENT
- **Verbal Sync**: What was being said? Was gesture BEFORE/DURING/AFTER the word?
- **Touch Target**: SKIN / HAIR / OBJECT (hat, headphones, glasses) / N/A
- **Significance**: IGNORE / LOW / MEDIUM / HIGH / CRITICAL

FACS QUICK REFERENCE:
AU1+AU2: Surprise | AU4: Anger/Concentration | AU6+AU12: Genuine smile
AU12: Social smile | AU14: Contempt | AU15: Sadness | AU17: Doubt | AU24: Tension

**GEAR ADJUSTMENT FILTER (Critical - Apply BEFORE scoring):**

STEP 1: Identify all gear the subject is wearing:
- Headphones/earbuds? → Ear/head touches may be INSTRUMENTAL
- Hat/cap? → Head touches may be INSTRUMENTAL
- Glasses? → Face touches near eyes may be INSTRUMENTAL
- Jewelry (rings, watch, necklace)? → Object manipulation may be INSTRUMENTAL

STEP 2: For each potential self-adaptor, check TOUCH TARGET:
- SKIN contact (cheek, nose, neck skin, lips) = PSYCHOLOGICAL - score normally
- OBJECT contact (headphone cup, hat brim, glasses frame) = INSTRUMENTAL - score 0

STEP 3: For OBJECT touches, only upgrade to psychological if:
- Touch is REPETITIVE (3+ times in 30 seconds) AND
- Touch occurs during SENSITIVE TOPICS (not random)

**GESTURAL-VERBAL SYNCHRONY (Include in every entry):**
- Gesture BEFORE word = AUTHENTIC (natural speech)
- Gesture DURING word = AUTHENTIC
- Gesture AFTER word = POSSIBLE REHEARSAL (flag as MISMATCH category)

**EXAMPLE LOG:**

| 0:34 | Single-shoulder shrug | - | 3 | MISMATCH | "definitely" - AFTER | N/A | HIGH - uncertainty leak |
| 0:52 | Nose touch | - | 2 | PACIFIER | "the deal" - DURING | SKIN | MEDIUM - skin contact |
| 1:12 | Hand to head | - | 0 | EQUIPMENT | casual speech | OBJECT (hat) | IGNORE - instrumental |
| 1:45 | Lip compression | AU24 | 2 | STRESS | "never said that" - DURING | N/A | HIGH - denial + tension |
| 2:03 | Head shake "no" | - | 3 | MISMATCH | "yes, absolutely" - DURING | N/A | CRITICAL - verbal contradiction |

=== SUMMARY STATISTICS ===

TOTAL EVENTS LOGGED: [count]
EVENTS FILTERED (Instrumental/Equipment): [count]
NET PSYCHOLOGICAL EVENTS: [count]

TOTAL BTE SCORE (psychological events only): [sum]
- Below 8: Low concern
- 8-12: Moderate concern
- 12+: High concern - investigate

GESTURAL MISMATCHES DETECTED: [count]
- List each with timestamp

STRESS CLUSTERS (3+ events within 5 sec):
- [Timestamp range]: [Events] - [Topic being discussed]

TOP 5 CRITICAL MOMENTS (ranked by significance):
1. [Time] - [Event] - [Why critical]
2. ...

=== GEAR FILTER AUDIT ===
Gear identified: [list]
Events filtered as instrumental: [count]
- [Time]: [Event] - [Reason filtered]

This is the SINGLE SOURCE OF TRUTH for all behavioral observations."""


DEEPFAKE_DETECTION_PROMPT = """Analyze video for deepfake/AI-generation artifacts.

**PURPOSE: Authentication layer before behavioral profiling.**
Digital-native subjects (influencers, crypto traders) may use AI-generated or manipulated content.

DETECTION CRITERIA:

1. FACIAL CONSISTENCY:
   - Does face lighting match background lighting?
   - Are there unnatural shadows or highlight patterns?
   - Does facial movement appear fluid or "floaty"?
   - Are there temporal glitches (face jumps/resets)?

2. AUDIO-VISUAL SYNC:
   - Do lip movements match audio precisely?
   - Are there micro-delays between speech and mouth movement?
   - Does breath timing match visible chest movement?

3. BOUNDARY ARTIFACTS:
   - Is there blurring/shimmer at face-hair boundary?
   - Are earlobes/ears rendered consistently?
   - Does neck-face boundary show artifacts?
   - Are teeth rendered consistently across frames?

4. TEMPORAL COHERENCE:
   - Does blink pattern appear natural or regularized?
   - Are there impossible physiological movements?
   - Do micro-expressions appear or disappear too sharply?

5. BACKGROUND ANALYSIS:
   - Does background show warping during face movement?
   - Are reflective surfaces consistent with face position?
   - Do shadows move appropriately with subject movement?

6. AUDIO ARTIFACTS:
   - Unnatural pauses or breathing patterns
   - Clipped phonemes or synthetic-sounding segments
   - Room reverb inconsistencies

**OUTPUT FORMAT:**

=== AUTHENTICITY ASSESSMENT ===

OVERALL VERDICT: AUTHENTIC / LIKELY AUTHENTIC / SUSPICIOUS / LIKELY SYNTHETIC

ARTIFACT ANALYSIS:
| Category | Finding | Confidence | Timestamp |
|----------|---------|------------|-----------|
| Facial consistency | [observation] | HIGH/MED/LOW | [if applicable] |
| ... | ... | ... | ... |

RED FLAGS DETECTED: [count]
[List any suspicious indicators]

CONFIDENCE IN ASSESSMENT: HIGH / MEDIUM / LOW

RECOMMENDATION:
- AUTHENTIC: Proceed with behavioral analysis
- LIKELY AUTHENTIC: Proceed with caution, note in report
- SUSPICIOUS: Flag for human review before profiling
- LIKELY SYNTHETIC: Abort behavioral profiling, report as manipulated media

NOTE: This is a heuristic check, not forensic-grade detection.
High-quality deepfakes may pass this screening."""


AUDIO_LINGUISTIC_COMBINED_PROMPT = """Perform COMBINED audio-linguistic analysis (voice prosody + speech content).

**CRITICAL: YOU ARE ANALYZING RAW AUDIO, NOT A TEXT TRANSCRIPT.**
Listen to the actual audio waveform. DO NOT guess prosody from punctuation or word choice.
If you cannot hear the audio, explicitly state: "AUDIO MODALITY UNAVAILABLE - prosodic analysis limited."

**PURPOSE: Unified analysis of HOW it sounds and WHAT is said.**

=== PART A: PROSODIC ANALYSIS (From LISTENING to audio) ===

**YOU MUST LISTEN TO THE AUDIO FOR THIS SECTION.**

1. PITCH ANALYSIS (from audio waveform):
   - Estimated pitch floor: [Hz or Low/Mid/High]
   - Estimated pitch ceiling: [Hz or Low/Mid/High]
   - Pitch variability: MONOTONE / NORMAL / EXPRESSIVE / ERRATIC
   - Stress-induced pitch spikes: [timestamps] - describe what you HEARD

2. VOCAL FRY / CREAKY VOICE:
   - Onset timestamps: [when vocal fry begins]
   - Correlation with topics: [what was being discussed]
   - Interpretation: Fatigue? Affectation? Stress?

3. TEMPO & RHYTHM (from audio):
   - Speaking rate estimate: [WPM]
   - Acceleration points: [timestamps] - subject speeds up
   - Deceleration points: [timestamps] - subject slows down
   - Unnatural pauses: [timestamps] - strategic vs processing

4. VOICE QUALITY (from audio):
   - Baseline quality: Clear / Raspy / Breathy / Tense / Nasal
   - Quality shifts: [timestamps where voice quality changed]
   - Throat clearing/swallowing: [timestamps]
   - Micro-tremors detected: [timestamps] - vocal stress markers

5. RESPONSE LATENCY (from audio):
   - Measure time between question/prompt and response start
   - Flag delays >2 seconds with topic context

=== PART B: DOMAIN-SPECIFIC LEXICON (Critical for interpretation) ===

**STEP 1: IDENTIFY SUBJECT'S SUBCULTURE**
Based on vocabulary, topics, and presentation, identify primary domain:
- Crypto/Trading: "nuke," "rekt," "moon," "diamond hands," "cope," "ngmi"
- Gaming/Streaming: "pog," "based," "cringe," "meta," "buff/nerf"
- Tech/Startup: "ship," "pivot," "scale," "runway," "burn rate"
- Finance/Business: "leverage," "exposure," "hedge," "position"
- Military/LEO: "sitrep," "oscar mike," "tango," "copy that"
- Academic: formal register, citations, hedging language
- Other: [describe]

**STEP 2: RECALIBRATE EMOTIONAL WEIGHT**
Words that sound emotionally charged but are NEUTRAL in subject's domain:
| Word | Literal Meaning | Domain Meaning | Recalibrated Weight |
| "disgusting" | visceral rejection | bearish chart (trading) | NEUTRAL |
| "nuke" | destroy | sharp price drop | NEUTRAL |
| "kill" | end life | perform well | NEUTRAL |

**List domain-specific terms detected and their recalibrated interpretation.**

=== PART C: LINGUISTIC ANALYSIS (What Is Said) ===

6. SOCIOLINGUISTIC PROFILE:
   - Accent: [Regional/ethnic markers]
   - Register: FORMAL / CASUAL / TECHNICAL / STREET
   - Code-switching: [instances of register shifts]
   - Education markers: [vocabulary level]

7. COGNITIVE LOAD INDICATORS:
   - Syntax breakdown points: [where sentence structure collapsed]
   - Specificity→Philosophy pivots: [where facts became vague]
   - Verbal processing overload: [unusual word choices, repairs]

8. DECEPTION LINGUISTICS:
   - Minimizing language: [instances with timestamps]
   - Distancing language: [passive voice, "one might"]
   - Denial intensifiers: ["I swear," "To be honest"]

9. PRONOUN DRIFT + VELOCITY:
   Track pronoun shifts AND measure how fast they occur:
   | Timestamp | Before | After | Trigger | Velocity |

   VELOCITY CLASSIFICATION:
   - INSTANT (<0.5 sec after negative stimulus): Possibly rehearsed defense
   - GRADUAL (2-5 sec): Real-time cognitive processing
   - DELAYED (>5 sec): May indicate genuine reflection

   Example:
   | 1:23 | "I made the decision" | "The decision was made" | criticism | INSTANT - rehearsed |

=== PART D: GAZE INTERPRETATION (Screen Context) ===

**IF SUBJECT IS IDENTIFIED AS TRADER/ENGINEER/DATA WORKER:**
Reinterpret "Up-Right" gaze patterns:
- Standard interpretation: Visual Constructed (fabrication)
- Domain-adjusted: Data Visualization / Chart Access / Mental calculation
- Only flag as deceptive if:
  - Gaze occurs during SIMPLE questions (not data discussion)
  - Gaze is accompanied by other stress clusters

=== INTEGRATED CREDIBILITY ASSESSMENT ===

AUDIO MODALITY STATUS: AVAILABLE / UNAVAILABLE
(If unavailable, prosodic analysis is limited to inference - flag this clearly)

DOMAIN-ADJUSTED CREDIBILITY: HIGH / MODERATE / LOW

TOP 3 CONCERN POINTS (after domain adjustment):
1. [Timestamp] - [What was said] + [How it sounded] = [Interpretation]
2. ...
3. ...

PRONOUN DRIFT VELOCITY SUMMARY:
- Total shifts detected: [count]
- INSTANT shifts (possible rehearsal): [count]
- Topics with highest drift: [list]

ALTERNATIVE EXPLANATIONS:
- [Non-deceptive reasons for observed patterns]"""


# =============================================================================
# PROMPT GROUPS FOR PARALLEL EXECUTION
# Core prompts + NCI/Chase Hughes behavioral analysis prompts
# =============================================================================

# Stage 0 prompts - run FIRST before main analysis
STAGE_ZERO_PROMPTS = {
    'subject_identification': SUBJECT_IDENTIFICATION_PROMPT,
    'baseline_establishment': BASELINE_ESTABLISHMENT_PROMPT,
    'deepfake_detection': DEEPFAKE_DETECTION_PROMPT,
}

VISUAL_PROMPTS = {
    # KINESIC EVENT LOG - single source of truth for ALL behavioral observations
    # Consolidates: FACS, BTE, Body Language, Gestural Mismatch, Gesture Timeline
    'kinesic_log': KINESIC_EVENT_LOG_PROMPT,
    # Core visual analysis
    'archetype': VISUAL_ARCHETYPE_PROMPT,
    'congruence': VISUAL_CONGRUENCE_PROMPT,
    # CV-validated blink rate (ground truth)
    'blink_rate': VISUAL_BLINK_RATE_PROMPT,
    # Removed (consolidated into kinesic_log):
    # - 'facs', 'body_language', 'bte_scoring', 'stress_clusters', 'gestural_mismatch'
    # Removed (moved to Stage 0):
    # - 'subject_identification'
    # Removed (low validity):
    # - 'facial_etching'
}

MULTIMODAL_PROMPTS = {
    # Core multimodal analysis
    'cross_modal_sync': MULTIMODAL_CROSS_MODAL_SYNC_PROMPT,
    'environment': MULTIMODAL_ENVIRONMENT_PROMPT,
    'camera_awareness': MULTIMODAL_CAMERA_AWARENESS_PROMPT,
    # NCI/Chase Hughes additions
    'five_cs': MULTIMODAL_FIVE_CS_PROMPT,
    # Removed (consolidated into kinesic_log):
    # - 'gesture_timeline'
    # Removed (moved to Stage 0):
    # - 'baseline_deviation' (now baseline_establishment in Stage 0)
}

AUDIO_PROMPTS = {
    # COMBINED audio-linguistic (merges voice + sociolinguistic + credibility)
    'audio_linguistic': AUDIO_LINGUISTIC_COMBINED_PROMPT,
    # NCI/Chase Hughes additions (kept separate - distinct analysis types)
    'detail_mountain_valley': AUDIO_DETAIL_MOUNTAIN_VALLEY_PROMPT,
    'minimizing_language': AUDIO_MINIMIZING_LANGUAGE_PROMPT,
    'linguistic_harvesting': AUDIO_LINGUISTIC_HARVESTING_PROMPT,
    # LIWC quantitative analysis
    'liwc': AUDIO_LIWC_PROMPT,
    # Removed (consolidated into audio_linguistic):
    # - 'voice_characteristics': Merged into audio_linguistic
    # - 'sociolinguistic': Merged into audio_linguistic
    # - 'credibility': Merged into audio_linguistic
}

SYNTHESIS_PROMPTS = {
    # Core synthesis analysis
    'personality': SYNTHESIS_PERSONALITY_PROMPT,
    'threat': SYNTHESIS_THREAT_PROMPT,
    'differential': SYNTHESIS_DIFFERENTIAL_PROMPT,
    'contradictions': SYNTHESIS_CONTRADICTIONS_PROMPT,
    'red_team': SYNTHESIS_RED_TEAM_PROMPT,
    # Internet culture filter (runs before NCI for recalibration)
    'irony_meme_filter': SYNTHESIS_IRONY_MEME_FILTER_PROMPT,
    # NCI/Chase Hughes additions
    'fate_model': SYNTHESIS_FATE_MODEL_PROMPT,
    'nci_deception_summary': SYNTHESIS_NCI_DECEPTION_SUMMARY_PROMPT,
    # Final integration (must be last)
    'final': SYNTHESIS_FINAL_INTEGRATION_PROMPT,
}

# =============================================================================
# CASE FILE SYNTHESIS PROMPT (Replaces FBI final integration)
# =============================================================================

CASE_FILE_SYNTHESIS_PROMPT = """Generate a structured FBI-style CASE FILE based on all analyses.

You are creating an official investigative case file. This must be formatted as a professional dossier.

**SUBJECT INFORMATION:**
{subject_info}

**ALL ANALYSIS DATA:**
{previous_analyses}

**SYNTHESIS SUB-ANALYSES:**
{synthesis_results}

═══════════════════════════════════════════════════════════════════════════════
                        BEHAVIORAL ANALYSIS UNIT
                           CASE FILE REPORT
═══════════════════════════════════════════════════════════════════════════════

CASE ID: {case_id}
DATE: {date}
CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY
ANALYST: Automated Behavioral Analysis System

═══════════════════════════════════════════════════════════════════════════════
SECTION 1: SUBJECT IDENTIFICATION
═══════════════════════════════════════════════════════════════════════════════

[Insert subject identification from subject_info - name if known, or physical description]
[Include any known aliases, online handles, platform presence]

═══════════════════════════════════════════════════════════════════════════════
SECTION 2: EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

[2-3 sentence overview of who this subject is and what the behavioral analysis reveals]
[Primary behavioral classification]
[Key finding in one sentence]

═══════════════════════════════════════════════════════════════════════════════
SECTION 3: THREAT ASSESSMENT
═══════════════════════════════════════════════════════════════════════════════

OVERALL THREAT LEVEL: [NONE / LOW / MODERATE / HIGH / SEVERE]

Risk Categories:
- Physical Threat: [Level] - [Brief justification]
- Financial/Fraud Risk: [Level] - [Brief justification]
- Manipulation Risk: [Level] - [Brief justification]
- Flight Risk: [Level] - [Brief justification]

═══════════════════════════════════════════════════════════════════════════════
SECTION 4: PSYCHOLOGICAL PROFILE
═══════════════════════════════════════════════════════════════════════════════

PRIMARY BEHAVIORAL ARCHETYPE: [From archetype analysis]
SECONDARY TRAITS: [Supporting characteristics]

BIG FIVE PROFILE:
[Include scores with brief behavioral evidence]

DARK TRIAD ASSESSMENT:
[Include scores with behavioral evidence]

MBTI HYPOTHESIS: [Type] - [Confidence level]

═══════════════════════════════════════════════════════════════════════════════
SECTION 5: DECEPTION ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

OVERALL VERACITY ASSESSMENT: [TRUTHFUL / MIXED / DECEPTIVE]

DECEPTION INDICATORS DETECTED:
[List key moments with timestamps where deception markers appeared]

COGNITIVE LOAD HOTSPOTS:
[List topics/moments that produced highest cognitive stress]

BODY-VERBAL CONFLICTS:
[Table of key conflicts between what was said vs body language]

═══════════════════════════════════════════════════════════════════════════════
SECTION 6: OPERATIONAL RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

INTERVIEW APPROACH: [Recommended strategy]
PSYCHOLOGICAL LEVERAGE POINTS: [What can be used]
DE-ESCALATION TRIGGERS: [What to avoid]
PREDICTED RESPONSES: [How subject likely reacts to pressure]

═══════════════════════════════════════════════════════════════════════════════
SECTION 7: INVESTIGATIVE PRIORITIES
═══════════════════════════════════════════════════════════════════════════════

[List specific areas requiring further investigation]
[Flag any statements that warrant verification]
[Note any associates mentioned or implied]

═══════════════════════════════════════════════════════════════════════════════
                           END OF CASE FILE
═══════════════════════════════════════════════════════════════════════════════

CRITICAL WEIGHTING (Apply throughout):
- Body language signals carry 5x weight of verbal content
- Involuntary behaviors outweigh voluntary presentations
- Cross-modal conflicts are primary deception indicators

Generate the complete case file following this exact structure."""

