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
# NCI/CHASE HUGHES VISUAL SUB-PROMPTS (Stage 3 - run in parallel)
# Based on The Behavior Ops Manual and Six-Minute X-Ray methodologies
# =============================================================================

VISUAL_BLINK_RATE_PROMPT = """Analyze blink rate patterns as stress and deception indicators.

BASELINE REFERENCE (Chase Hughes/NCI methodology):
- Normal baseline: 17-25 blinks per minute (BPM)
- Focused/calm state: 7-10 BPM (reduced blinking indicates engagement)
- Stressed/deceptive state: Up to 50 BPM (elevated blinking indicates stress)

Analyze the subject's blinking patterns throughout the video:

1. BASELINE ESTIMATION:
   - Estimate the subject's baseline blink rate during neutral moments
   - Note any periods of sustained low blink rate (focus/calm)
   - Note any periods of elevated blink rate (stress/anxiety)

2. BLINK RATE CHANGES:
   - Identify moments where blink rate INCREASES suddenly
   - Note what topic or question preceded the change
   - Flag these as potential stress indicators

3. TOPIC-CORRELATED ANALYSIS:
   - Which topics or statements correlated with elevated blinking?
   - Which topics showed reduced/calm blinking?
   - Create a blink rate timeline if possible

4. DECEPTION CORRELATION:
   - Elevated blink rate alone does NOT prove deception
   - Note as STRESS INDICATOR requiring follow-up
   - Correlate with other behavioral clusters

Provide:
- Estimated baseline blink rate: [X] BPM
- Peak elevated rate observed: [X] BPM
- Moments of elevated blinking: [timestamp/topic]
- Blink rate assessment: NORMAL / ELEVATED / HIGHLY ELEVATED
- Stress correlation confidence: Low/Medium/High

Focus ONLY on blink rate analysis - be precise with observations."""


VISUAL_BTE_SCORING_PROMPT = """Score behavioral indicators using the Behavioral Table of Elements (BTE) methodology.

CHASE HUGHES BTE SCORING SYSTEM:
Score individual stress/deception indicators. A cumulative score of 12+ suggests likely deception.

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


VISUAL_GESTURAL_MISMATCH_PROMPT = """Detect gestural mismatches and timing asynchrony as deception indicators.

CHASE HUGHES GESTURAL-VERBAL SYNCHRONY ANALYSIS:
Gestural-verbal asynchrony is one of the highest indicators of rehearsed deception.

Analyze for these specific mismatches:

1. GESTURAL-VERBAL TIMING:
   - Natural speech: Gesture PRECEDES or COINCIDES with verbal emphasis
   - Rehearsed/deceptive: Gesture FOLLOWS after the emphasized word
   - Example: Hand chop on "opportunity" - does gesture hit BEFORE/DURING or AFTER the word?

2. HEAD SHAKE MISMATCH:
   - Subject says positive statement while shaking head "no"
   - Subject says negative statement while nodding "yes"
   - Flag specific instances with timestamps

3. SHOULDER SHRUG MISMATCH:
   - Single-shoulder shrug during definitive statements (HIGH deception indicator)
   - Full shoulder shrug with uncertain language (congruent)
   - Single-shoulder = "I don't really believe what I'm saying"

4. FACIAL-VERBAL MISMATCH:
   - Smiling while discussing negative topics
   - Flat affect while expressing enthusiasm
   - Contempt flash during positive statements

5. TIMING ANALYSIS:
   - Delayed emotional reactions (>0.5 sec after trigger)
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
# Core prompts + NCI/Chase Hughes behavioral analysis prompts
# =============================================================================

VISUAL_PROMPTS = {
    # Core visual analysis
    'facs': VISUAL_FACS_PROMPT,
    'archetype': VISUAL_ARCHETYPE_PROMPT,
    'body_language': VISUAL_BODY_LANGUAGE_PROMPT,
    'deception': VISUAL_DECEPTION_PROMPT,
    # NCI/Chase Hughes additions
    'blink_rate': VISUAL_BLINK_RATE_PROMPT,
    'bte_scoring': VISUAL_BTE_SCORING_PROMPT,
    'facial_etching': VISUAL_FACIAL_ETCHING_PROMPT,
    'gestural_mismatch': VISUAL_GESTURAL_MISMATCH_PROMPT,
    'stress_clusters': VISUAL_STRESS_CLUSTER_PROMPT,
}

MULTIMODAL_PROMPTS = {
    # Core multimodal analysis
    'gesture_timeline': MULTIMODAL_GESTURE_TIMELINE_PROMPT,
    'cross_modal_sync': MULTIMODAL_CROSS_MODAL_SYNC_PROMPT,
    'environment': MULTIMODAL_ENVIRONMENT_PROMPT,
    'camera_awareness': MULTIMODAL_CAMERA_AWARENESS_PROMPT,
    # NCI/Chase Hughes additions
    'five_cs': MULTIMODAL_FIVE_CS_PROMPT,
    'baseline_deviation': MULTIMODAL_BASELINE_DEVIATION_PROMPT,
}

AUDIO_PROMPTS = {
    # Core audio analysis
    'voice_characteristics': AUDIO_VOICE_CHARACTERISTICS_PROMPT,
    'sociolinguistic': AUDIO_SOCIOLINGUISTIC_PROMPT,
    'deception_voice': AUDIO_DECEPTION_VOICE_PROMPT,
    # NCI/Chase Hughes additions
    'detail_mountain_valley': AUDIO_DETAIL_MOUNTAIN_VALLEY_PROMPT,
    'minimizing_language': AUDIO_MINIMIZING_LANGUAGE_PROMPT,
    'linguistic_harvesting': AUDIO_LINGUISTIC_HARVESTING_PROMPT,
}

SYNTHESIS_PROMPTS = {
    # Core synthesis analysis
    'personality': SYNTHESIS_PERSONALITY_PROMPT,
    'threat': SYNTHESIS_THREAT_PROMPT,
    'differential': SYNTHESIS_DIFFERENTIAL_PROMPT,
    'contradictions': SYNTHESIS_CONTRADICTIONS_PROMPT,
    'red_team': SYNTHESIS_RED_TEAM_PROMPT,
    # NCI/Chase Hughes additions
    'fate_model': SYNTHESIS_FATE_MODEL_PROMPT,
    'nci_deception_summary': SYNTHESIS_NCI_DECEPTION_SUMMARY_PROMPT,
    # Final integration (must be last)
    'final': SYNTHESIS_FINAL_INTEGRATION_PROMPT,
}
