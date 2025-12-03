"""
Programmatic LIWC-Style Linguistic Analysis Module.
Performs actual word counting and categorization using NLP,
then passes verified data to LLM for psychological interpretation.

This replaces LLM-simulated word counting with ground-truth analysis.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import spaCy for advanced NLP
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logger.warning("spaCy not installed. Using regex-based analysis. Install with: pip install spacy")


@dataclass
class LIWCCategory:
    """LIWC category with word lists and counts."""
    name: str
    description: str
    words: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    count: int = 0
    examples_found: List[str] = field(default_factory=list)


# LIWC-22 Inspired Categories with comprehensive word lists
LIWC_CATEGORIES = {
    # ==================== PRONOUN CATEGORIES ====================
    "i_words": LIWCCategory(
        name="First Person Singular (I)",
        description="Self-references indicating self-focus, potentially narcissism or anxiety",
        words=["i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd", "im"],
        patterns=[r"\bi['']m\b", r"\bi['']ve\b", r"\bi['']ll\b", r"\bi['']d\b"]
    ),
    "we_words": LIWCCategory(
        name="First Person Plural (We)",
        description="Group identity, collective thinking, social cohesion",
        words=["we", "us", "our", "ours", "ourselves", "we're", "we've", "we'll", "we'd"],
        patterns=[r"\bwe['']re\b", r"\bwe['']ve\b", r"\bwe['']ll\b"]
    ),
    "you_words": LIWCCategory(
        name="Second Person (You)",
        description="Direct address, engagement, persuasion attempts",
        words=["you", "your", "yours", "yourself", "yourselves", "you're", "you've", "you'll", "you'd"],
        patterns=[r"\byou['']re\b", r"\byou['']ve\b", r"\byou['']ll\b"]
    ),
    "they_words": LIWCCategory(
        name="Third Person (They/He/She)",
        description="Narrative distance, social awareness, external focus",
        words=["he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
               "himself", "herself", "themselves", "he's", "she's", "they're", "they've"],
        patterns=[r"\bhe['']s\b", r"\bshe['']s\b", r"\bthey['']re\b", r"\bthey['']ve\b"]
    ),

    # ==================== EMOTIONAL CATEGORIES ====================
    "positive_emotion": LIWCCategory(
        name="Positive Emotion",
        description="Positive affect, happiness, optimism",
        words=["happy", "love", "great", "good", "nice", "excellent", "wonderful", "amazing",
               "fantastic", "beautiful", "joy", "excited", "thrilled", "delighted", "pleased",
               "glad", "cheerful", "content", "satisfied", "proud", "hopeful", "grateful",
               "fortunate", "lucky", "blessed", "awesome", "brilliant", "perfect", "best",
               "fun", "enjoy", "like", "adore", "appreciate", "value", "cherish", "treasure"]
    ),
    "negative_emotion": LIWCCategory(
        name="Negative Emotion",
        description="Negative affect, distress, problems",
        words=["hate", "bad", "terrible", "awful", "horrible", "sad", "angry", "upset",
               "frustrated", "annoyed", "irritated", "furious", "enraged", "depressed",
               "miserable", "unhappy", "disappointed", "worried", "anxious", "scared",
               "afraid", "fearful", "nervous", "stressed", "tense", "hurt", "pain",
               "suffer", "struggle", "fail", "wrong", "problem", "trouble", "difficult",
               "hard", "tough", "rough", "ugly", "disgusting", "sick", "tired", "exhausted"]
    ),
    "anxiety_words": LIWCCategory(
        name="Anxiety",
        description="Fear, worry, nervousness - potential stress indicators",
        words=["worried", "nervous", "anxious", "afraid", "scared", "fear", "panic",
               "stress", "tense", "uneasy", "concerned", "apprehensive", "dread",
               "terrified", "frightened", "alarmed", "distressed", "restless", "jittery",
               "on edge", "overwhelmed", "paranoid", "insecure", "uncertain", "doubtful"]
    ),
    "anger_words": LIWCCategory(
        name="Anger",
        description="Hostility, aggression, frustration",
        words=["angry", "mad", "furious", "rage", "hate", "hostile", "aggressive",
               "irritated", "annoyed", "frustrated", "pissed", "livid", "outraged",
               "resentful", "bitter", "vengeful", "violent", "attack", "fight", "destroy",
               "kill", "murder", "revenge", "damn", "hell", "crap", "stupid", "idiot"]
    ),
    "sadness_words": LIWCCategory(
        name="Sadness",
        description="Depression, grief, hopelessness",
        words=["sad", "depressed", "unhappy", "miserable", "grief", "sorrow", "crying",
               "tears", "lonely", "alone", "empty", "hopeless", "despair", "heartbroken",
               "devastated", "crushed", "hurt", "pain", "loss", "missing", "regret",
               "sorry", "apologize", "guilt", "shame", "worthless", "helpless"]
    ),

    # ==================== COGNITIVE CATEGORIES ====================
    "certainty_words": LIWCCategory(
        name="Certainty",
        description="Confidence, definiteness - high certainty may indicate deception",
        words=["always", "never", "definitely", "certainly", "absolutely", "completely",
               "totally", "entirely", "undoubtedly", "surely", "clearly", "obviously",
               "of course", "without doubt", "guaranteed", "positive", "certain", "sure",
               "know", "fact", "truth", "proven", "evident", "must", "will", "every",
               "all", "none", "nothing", "everything", "everyone", "nobody"]
    ),
    "tentative_words": LIWCCategory(
        name="Tentative",
        description="Hedging, uncertainty, vagueness - may indicate honesty or evasion",
        words=["maybe", "perhaps", "possibly", "might", "could", "would", "should",
               "seem", "appear", "guess", "think", "believe", "suppose", "assume",
               "probably", "likely", "unlikely", "sort of", "kind of", "somewhat",
               "rather", "fairly", "quite", "almost", "nearly", "about", "around",
               "approximately", "roughly", "generally", "usually", "often", "sometimes"]
    ),
    "causation_words": LIWCCategory(
        name="Causation",
        description="Cause-effect reasoning, logical thinking",
        words=["because", "since", "therefore", "hence", "thus", "so", "cause",
               "effect", "result", "consequence", "reason", "why", "lead", "produce",
               "create", "make", "force", "enable", "allow", "prevent", "stop",
               "due to", "owing to", "thanks to", "as a result", "consequently"]
    ),
    "insight_words": LIWCCategory(
        name="Insight",
        description="Self-reflection, understanding, awareness",
        words=["think", "know", "understand", "realize", "recognize", "see", "feel",
               "sense", "believe", "consider", "reflect", "ponder", "wonder", "imagine",
               "discover", "learn", "find", "notice", "observe", "aware", "conscious",
               "insight", "revelation", "epiphany", "clarity", "comprehend", "grasp"]
    ),

    # ==================== SOCIAL CATEGORIES ====================
    "social_words": LIWCCategory(
        name="Social",
        description="Social processes, relationships, interaction",
        words=["friend", "family", "people", "person", "group", "team", "community",
               "society", "relationship", "partner", "colleague", "neighbor", "member",
               "talk", "speak", "say", "tell", "ask", "answer", "discuss", "meet",
               "join", "share", "help", "support", "together", "social", "public"]
    ),
    "family_words": LIWCCategory(
        name="Family",
        description="Family references, domestic relationships",
        words=["family", "mother", "father", "mom", "dad", "parent", "child", "children",
               "son", "daughter", "brother", "sister", "sibling", "grandma", "grandpa",
               "grandmother", "grandfather", "aunt", "uncle", "cousin", "nephew", "niece",
               "husband", "wife", "spouse", "married", "wedding", "home", "house"]
    ),
    "money_words": LIWCCategory(
        name="Money/Finance",
        description="Financial references - important in fraud profiling",
        words=["money", "cash", "dollar", "price", "cost", "pay", "paid", "buy", "sell",
               "profit", "loss", "invest", "investment", "stock", "market", "business",
               "deal", "offer", "contract", "fee", "charge", "expense", "income", "salary",
               "rich", "wealthy", "poor", "afford", "budget", "bank", "credit", "debt",
               "loan", "finance", "financial", "economic", "economy", "crypto", "bitcoin"]
    ),
    "power_words": LIWCCategory(
        name="Power/Status",
        description="Dominance, authority, status - narcissism indicators",
        words=["power", "control", "lead", "leader", "boss", "chief", "authority",
               "command", "order", "rule", "dominate", "superior", "important", "famous",
               "successful", "winner", "best", "top", "elite", "exclusive", "special",
               "unique", "exceptional", "extraordinary", "impressive", "influence",
               "prestigious", "luxury", "expensive", "brand", "status", "vip"]
    ),

    # ==================== DECEPTION INDICATORS ====================
    "negation_words": LIWCCategory(
        name="Negation",
        description="Denial, negation - elevated use may indicate deception",
        words=["no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither",
               "nor", "without", "cannot", "can't", "won't", "wouldn't", "couldn't",
               "shouldn't", "didn't", "doesn't", "don't", "isn't", "aren't", "wasn't",
               "weren't", "hasn't", "haven't", "hadn't"]
    ),
    "exclusive_words": LIWCCategory(
        name="Exclusive",
        description="Exclusion, distinction - complex narratives, potential deception",
        words=["but", "except", "without", "exclude", "only", "just", "however",
               "although", "though", "unless", "rather", "instead", "besides",
               "otherwise", "nevertheless", "nonetheless", "yet", "still", "whereas"]
    ),
    "filler_words": LIWCCategory(
        name="Fillers",
        description="Hesitation markers - may indicate cognitive load or deception",
        words=["um", "uh", "er", "ah", "like", "you know", "i mean", "basically",
               "actually", "literally", "honestly", "frankly", "well", "so", "anyway",
               "right", "okay", "ok", "yeah", "yep", "nah", "hmm", "huh"]
    ),

    # ==================== TEMPORAL CATEGORIES ====================
    "past_focus": LIWCCategory(
        name="Past Focus",
        description="Past tense, historical references",
        words=["was", "were", "had", "did", "been", "ago", "before", "yesterday",
               "last", "previous", "former", "once", "used to", "back then", "remember",
               "recalled", "happened", "occurred", "ended", "finished", "completed"]
    ),
    "present_focus": LIWCCategory(
        name="Present Focus",
        description="Present tense, current state",
        words=["is", "are", "am", "be", "being", "now", "today", "currently",
               "presently", "at the moment", "right now", "these days", "lately",
               "recently", "happening", "ongoing", "existing", "living"]
    ),
    "future_focus": LIWCCategory(
        name="Future Focus",
        description="Future tense, planning, anticipation",
        words=["will", "shall", "going to", "gonna", "tomorrow", "next", "soon",
               "later", "eventually", "future", "upcoming", "planned", "expect",
               "anticipate", "predict", "hope", "wish", "intend", "plan"]
    )
}


@dataclass
class LIWCAnalysisResult:
    """Complete LIWC analysis result with verified counts."""
    total_words: int
    total_sentences: int
    categories: Dict[str, Dict]
    pronoun_ratio: Dict[str, float]
    emotional_tone: Dict[str, float]
    cognitive_complexity: float
    deception_indicators: Dict[str, float]
    temporal_orientation: Dict[str, float]
    raw_text: str
    summary: str


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words."""
    # Convert to lowercase and split
    text = text.lower()
    # Keep contractions together, split on whitespace and punctuation
    words = re.findall(r"\b[\w']+\b", text)
    return words


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def analyze_category(words: List[str], category: LIWCCategory) -> Tuple[int, List[str]]:
    """
    Count words matching a LIWC category.

    Returns:
        Tuple of (count, examples_found)
    """
    count = 0
    examples = []
    word_lower_set = set(words)

    # Check direct word matches
    for target_word in category.words:
        target_lower = target_word.lower()
        matches = words.count(target_lower)
        if matches > 0:
            count += matches
            if target_lower not in examples:
                examples.append(target_lower)

    # Check pattern matches (for contractions, etc.)
    full_text = ' '.join(words)
    for pattern in category.patterns:
        pattern_matches = re.findall(pattern, full_text, re.IGNORECASE)
        count += len(pattern_matches)
        examples.extend([m.lower() for m in pattern_matches if m.lower() not in examples])

    return count, examples[:10]  # Limit examples to 10


def calculate_cognitive_complexity(category_results: Dict) -> float:
    """
    Calculate cognitive complexity score based on linguistic markers.
    Higher scores indicate more complex, analytical thinking.
    """
    # Factors that increase complexity
    insight_score = category_results.get('insight_words', {}).get('percentage', 0)
    causation_score = category_results.get('causation_words', {}).get('percentage', 0)
    tentative_score = category_results.get('tentative_words', {}).get('percentage', 0)
    exclusive_score = category_results.get('exclusive_words', {}).get('percentage', 0)

    # Calculate weighted complexity
    complexity = (
        insight_score * 0.3 +
        causation_score * 0.3 +
        tentative_score * 0.2 +
        exclusive_score * 0.2
    )

    # Normalize to 0-100 scale
    return min(100, complexity * 10)


def calculate_deception_indicators(category_results: Dict, total_words: int) -> Dict[str, float]:
    """
    Calculate deception indicator scores based on linguistic research.

    Research shows deceptive speech often has:
    - Fewer first-person pronouns
    - More negative emotion words
    - Fewer exclusive words (simpler narratives)
    - More certainty words (overcompensation)
    """
    i_percentage = category_results.get('i_words', {}).get('percentage', 0)
    negative_percentage = category_results.get('negative_emotion', {}).get('percentage', 0)
    certainty_percentage = category_results.get('certainty_words', {}).get('percentage', 0)
    exclusive_percentage = category_results.get('exclusive_words', {}).get('percentage', 0)
    filler_percentage = category_results.get('filler_words', {}).get('percentage', 0)
    negation_percentage = category_results.get('negation_words', {}).get('percentage', 0)

    return {
        'low_self_reference': max(0, 5 - i_percentage) * 10,  # Low I-words suspicious
        'negative_affect': negative_percentage * 5,
        'overconfidence': certainty_percentage * 3,
        'narrative_simplicity': max(0, 3 - exclusive_percentage) * 10,
        'cognitive_load': filler_percentage * 5,
        'denial_frequency': negation_percentage * 3,
        'overall_risk': 0  # Calculated below
    }


def analyze_text_liwc(text: str) -> LIWCAnalysisResult:
    """
    Perform comprehensive LIWC-style analysis on text.

    Args:
        text: Raw text to analyze (transcript)

    Returns:
        LIWCAnalysisResult with verified counts and percentages
    """
    if not text or not text.strip():
        return LIWCAnalysisResult(
            total_words=0,
            total_sentences=0,
            categories={},
            pronoun_ratio={},
            emotional_tone={},
            cognitive_complexity=0,
            deception_indicators={},
            temporal_orientation={},
            raw_text=text,
            summary="No text provided for analysis."
        )

    # Tokenize
    words = tokenize_text(text)
    total_words = len(words)
    total_sentences = count_sentences(text)

    if total_words == 0:
        return LIWCAnalysisResult(
            total_words=0,
            total_sentences=0,
            categories={},
            pronoun_ratio={},
            emotional_tone={},
            cognitive_complexity=0,
            deception_indicators={},
            temporal_orientation={},
            raw_text=text,
            summary="No analyzable words found."
        )

    # Analyze each category
    category_results = {}
    for cat_key, category in LIWC_CATEGORIES.items():
        count, examples = analyze_category(words, category)
        percentage = (count / total_words) * 100 if total_words > 0 else 0

        category_results[cat_key] = {
            'name': category.name,
            'description': category.description,
            'count': count,
            'percentage': round(percentage, 2),
            'examples': examples
        }

    # Calculate pronoun ratios
    total_pronouns = sum([
        category_results['i_words']['count'],
        category_results['we_words']['count'],
        category_results['you_words']['count'],
        category_results['they_words']['count']
    ])

    pronoun_ratio = {
        'i_ratio': category_results['i_words']['count'] / total_pronouns if total_pronouns > 0 else 0,
        'we_ratio': category_results['we_words']['count'] / total_pronouns if total_pronouns > 0 else 0,
        'you_ratio': category_results['you_words']['count'] / total_pronouns if total_pronouns > 0 else 0,
        'they_ratio': category_results['they_words']['count'] / total_pronouns if total_pronouns > 0 else 0,
        'total_pronouns': total_pronouns,
        'pronoun_density': (total_pronouns / total_words) * 100 if total_words > 0 else 0
    }

    # Calculate emotional tone
    pos_count = category_results['positive_emotion']['count']
    neg_count = category_results['negative_emotion']['count']
    total_emotion = pos_count + neg_count

    emotional_tone = {
        'positive_percentage': category_results['positive_emotion']['percentage'],
        'negative_percentage': category_results['negative_emotion']['percentage'],
        'anxiety_percentage': category_results['anxiety_words']['percentage'],
        'anger_percentage': category_results['anger_words']['percentage'],
        'sadness_percentage': category_results['sadness_words']['percentage'],
        'emotional_balance': (pos_count - neg_count) / total_emotion if total_emotion > 0 else 0,
        'overall_emotionality': (total_emotion / total_words) * 100 if total_words > 0 else 0
    }

    # Calculate cognitive complexity
    cognitive_complexity = calculate_cognitive_complexity(category_results)

    # Calculate deception indicators
    deception_indicators = calculate_deception_indicators(category_results, total_words)
    # Calculate overall risk
    risk_factors = [v for k, v in deception_indicators.items() if k != 'overall_risk']
    deception_indicators['overall_risk'] = sum(risk_factors) / len(risk_factors) if risk_factors else 0

    # Calculate temporal orientation
    past = category_results['past_focus']['percentage']
    present = category_results['present_focus']['percentage']
    future = category_results['future_focus']['percentage']
    total_temporal = past + present + future

    temporal_orientation = {
        'past_focus': past,
        'present_focus': present,
        'future_focus': future,
        'dominant_orientation': 'past' if past >= present and past >= future else ('present' if present >= future else 'future'),
        'temporal_diversity': total_temporal
    }

    # Generate summary
    summary = generate_analysis_summary(
        total_words, total_sentences, category_results,
        pronoun_ratio, emotional_tone, cognitive_complexity,
        deception_indicators, temporal_orientation
    )

    return LIWCAnalysisResult(
        total_words=total_words,
        total_sentences=total_sentences,
        categories=category_results,
        pronoun_ratio=pronoun_ratio,
        emotional_tone=emotional_tone,
        cognitive_complexity=cognitive_complexity,
        deception_indicators=deception_indicators,
        temporal_orientation=temporal_orientation,
        raw_text=text,
        summary=summary
    )


def generate_analysis_summary(
    total_words: int,
    total_sentences: int,
    categories: Dict,
    pronoun_ratio: Dict,
    emotional_tone: Dict,
    cognitive_complexity: float,
    deception_indicators: Dict,
    temporal_orientation: Dict
) -> str:
    """Generate a human-readable summary of the LIWC analysis."""

    summary_parts = []

    # Basic stats
    summary_parts.append(f"LINGUISTIC STATISTICS:")
    summary_parts.append(f"  Total Words: {total_words}")
    summary_parts.append(f"  Total Sentences: {total_sentences}")
    summary_parts.append(f"  Avg Words/Sentence: {total_words/total_sentences:.1f}" if total_sentences > 0 else "  N/A")
    summary_parts.append("")

    # Pronoun analysis
    summary_parts.append("PRONOUN PROFILE:")
    i_pct = pronoun_ratio['i_ratio'] * 100
    we_pct = pronoun_ratio['we_ratio'] * 100
    summary_parts.append(f"  I-words: {categories['i_words']['count']} ({i_pct:.1f}% of pronouns)")
    summary_parts.append(f"  We-words: {categories['we_words']['count']} ({we_pct:.1f}% of pronouns)")

    # Interpret I-word usage
    if i_pct > 60:
        summary_parts.append("  >> HIGH self-focus detected - potential narcissism or anxiety indicator")
    elif i_pct < 20:
        summary_parts.append("  >> LOW self-reference - potential distancing or deception indicator")
    summary_parts.append("")

    # Emotional tone
    summary_parts.append("EMOTIONAL PROFILE:")
    summary_parts.append(f"  Positive Emotion: {emotional_tone['positive_percentage']:.2f}%")
    summary_parts.append(f"  Negative Emotion: {emotional_tone['negative_percentage']:.2f}%")
    summary_parts.append(f"  Anxiety Markers: {emotional_tone['anxiety_percentage']:.2f}%")
    summary_parts.append(f"  Anger Markers: {emotional_tone['anger_percentage']:.2f}%")

    balance = emotional_tone['emotional_balance']
    if balance > 0.3:
        summary_parts.append("  >> Predominantly POSITIVE emotional tone")
    elif balance < -0.3:
        summary_parts.append("  >> Predominantly NEGATIVE emotional tone")
    else:
        summary_parts.append("  >> MIXED/NEUTRAL emotional tone")
    summary_parts.append("")

    # Cognitive markers
    summary_parts.append("COGNITIVE PROFILE:")
    summary_parts.append(f"  Certainty Words: {categories['certainty_words']['count']} ({categories['certainty_words']['percentage']:.2f}%)")
    summary_parts.append(f"  Tentative Words: {categories['tentative_words']['count']} ({categories['tentative_words']['percentage']:.2f}%)")
    summary_parts.append(f"  Cognitive Complexity Score: {cognitive_complexity:.1f}/100")

    cert = categories['certainty_words']['percentage']
    tent = categories['tentative_words']['percentage']
    if cert > tent * 2:
        summary_parts.append("  >> HIGH certainty language - may indicate overconfidence or deception")
    elif tent > cert * 2:
        summary_parts.append("  >> HIGH hedging/tentative language - may indicate honesty or insecurity")
    summary_parts.append("")

    # Deception indicators
    summary_parts.append("DECEPTION RISK INDICATORS:")
    risk = deception_indicators['overall_risk']
    summary_parts.append(f"  Overall Deception Risk Score: {risk:.1f}/100")

    if deception_indicators['low_self_reference'] > 30:
        summary_parts.append("  >> FLAG: Low self-reference (distancing behavior)")
    if deception_indicators['overconfidence'] > 30:
        summary_parts.append("  >> FLAG: Excessive certainty language")
    if deception_indicators['narrative_simplicity'] > 30:
        summary_parts.append("  >> FLAG: Narrative lacks complexity")
    if deception_indicators['cognitive_load'] > 20:
        summary_parts.append("  >> FLAG: High filler word usage (cognitive load)")
    summary_parts.append("")

    # Money/Power (important for fraud)
    summary_parts.append("FRAUD-RELEVANT MARKERS:")
    summary_parts.append(f"  Money/Finance Words: {categories['money_words']['count']} ({categories['money_words']['percentage']:.2f}%)")
    summary_parts.append(f"  Power/Status Words: {categories['power_words']['count']} ({categories['power_words']['percentage']:.2f}%)")

    if categories['money_words']['percentage'] > 2 and categories['power_words']['percentage'] > 2:
        summary_parts.append("  >> ELEVATED focus on money and status - consistent with con artist profile")
    summary_parts.append("")

    # Temporal focus
    summary_parts.append("TEMPORAL ORIENTATION:")
    summary_parts.append(f"  Past Focus: {temporal_orientation['past_focus']:.2f}%")
    summary_parts.append(f"  Present Focus: {temporal_orientation['present_focus']:.2f}%")
    summary_parts.append(f"  Future Focus: {temporal_orientation['future_focus']:.2f}%")
    summary_parts.append(f"  Dominant: {temporal_orientation['dominant_orientation'].upper()}")

    return "\n".join(summary_parts)


def format_liwc_for_llm(result: LIWCAnalysisResult) -> str:
    """
    Format LIWC results for LLM interpretation.
    The LLM should interpret this verified data, NOT generate its own counts.
    """
    output = []
    output.append("=" * 60)
    output.append("VERIFIED LINGUISTIC ANALYSIS DATA")
    output.append("(Programmatic word counts - NOT LLM-generated)")
    output.append("=" * 60)
    output.append("")
    output.append(result.summary)
    output.append("")
    output.append("=" * 60)
    output.append("DETAILED CATEGORY COUNTS")
    output.append("=" * 60)

    for cat_key, cat_data in result.categories.items():
        if cat_data['count'] > 0:
            output.append(f"\n{cat_data['name']}:")
            output.append(f"  Count: {cat_data['count']} ({cat_data['percentage']:.2f}%)")
            if cat_data['examples']:
                output.append(f"  Examples: {', '.join(cat_data['examples'][:5])}")

    return "\n".join(output)
