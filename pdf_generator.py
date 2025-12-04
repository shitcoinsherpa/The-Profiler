"""
PDF Report Generator for FBI Behavioral Profiling System.
Generates professional PDF reports from analysis results.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


# Color scheme matching the FBI dark theme (only defined if reportlab is available)
if REPORTLAB_AVAILABLE:
    FBI_DARK_BLUE = colors.Color(0.039, 0.055, 0.102)  # #0a0e1a
    FBI_BLUE = colors.Color(0.118, 0.157, 0.259)  # #1e2842
    FBI_ACCENT = colors.Color(0.290, 0.620, 1.0)  # #4a9eff
    FBI_GOLD = colors.Color(1.0, 0.839, 0.039)  # #ffd60a
    FBI_TEXT = colors.Color(0.910, 0.918, 0.941)  # #e8eaf0
else:
    # Placeholder values when reportlab is not available
    FBI_DARK_BLUE = None
    FBI_BLUE = None
    FBI_ACCENT = None
    FBI_GOLD = None
    FBI_TEXT = None


def create_pdf_styles():
    """Create custom styles for the PDF report."""
    styles = getSampleStyleSheet()

    # Title style
    styles.add(ParagraphStyle(
        name='FBITitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=FBI_ACCENT,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))

    # Section header style
    styles.add(ParagraphStyle(
        name='FBISection',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=FBI_ACCENT,
        spaceBefore=20,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        borderPadding=(5, 5, 5, 5)
    ))

    # Subsection style
    styles.add(ParagraphStyle(
        name='FBISubsection',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.black,
        spaceBefore=15,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))

    # Body text style
    styles.add(ParagraphStyle(
        name='FBIBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        spaceBefore=4,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
        leading=14
    ))

    # Classification banner style
    styles.add(ParagraphStyle(
        name='FBIClassification',
        parent=styles['Normal'],
        fontSize=10,
        textColor=FBI_GOLD,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceBefore=0,
        spaceAfter=10
    ))

    # Metadata style
    styles.add(ParagraphStyle(
        name='FBIMeta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_LEFT,
        fontName='Helvetica'
    ))

    return styles


def generate_pdf_report(
    result: Dict,
    output_path: Optional[str] = None,
    subject_name: str = None
) -> str:
    """
    Generate a PDF report from analysis results.

    Args:
        result: Complete analysis result dictionary
        output_path: Optional output path (generates temp file if not provided)
        subject_name: Optional subject name to include in report

    Returns:
        Path to generated PDF file

    Raises:
        ImportError: If reportlab is not installed
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )

    # Set output path
    if output_path:
        pdf_path = Path(output_path)
    else:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.pdf',
            prefix='profile_report_'
        )
        pdf_path = Path(temp_file.name)
        temp_file.close()

    # Create document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = create_pdf_styles()
    story = []

    # Classification banner
    story.append(Paragraph(
        "CONFIDENTIAL - FOR RESEARCH USE ONLY",
        styles['FBIClassification']
    ))

    # Title
    story.append(Paragraph(
        "FBI-STYLE BEHAVIORAL PROFILE",
        styles['FBITitle']
    ))

    # Subject name if provided
    if subject_name:
        story.append(Paragraph(
            f"Subject: {subject_name}",
            styles['FBISection']
        ))
        story.append(Spacer(1, 10))

    # Case metadata table
    case_id = result.get('case_id', 'N/A')
    timestamp = result.get('timestamp', datetime.now().isoformat())
    processing_time = result.get('processing_time_seconds', 0)

    meta_data = [
        ['Case ID:', case_id],
        ['Date:', timestamp[:19] if len(timestamp) > 19 else timestamp],
        ['Processing Time:', f"{processing_time:.2f} seconds"],
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]

    meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)

    # Horizontal line
    story.append(Spacer(1, 15))
    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=FBI_ACCENT,
        spaceBefore=5,
        spaceAfter=15
    ))

    # Analyses
    analyses = result.get('analyses', {})

    # FBI Behavioral Synthesis (most important, first)
    if analyses.get('fbi_behavioral_synthesis'):
        story.append(Paragraph(
            "FBI BEHAVIORAL SYNTHESIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['fbi_behavioral_synthesis'], styles)
        story.append(PageBreak())

    # Sam Christensen Essence Profile
    if analyses.get('sam_christensen_essence'):
        story.append(Paragraph(
            "SAM CHRISTENSEN ESSENCE PROFILE",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['sam_christensen_essence'], styles)
        story.append(PageBreak())

    # Multimodal Behavioral Analysis
    if analyses.get('multimodal_behavioral'):
        story.append(Paragraph(
            "MULTIMODAL BEHAVIORAL ANALYSIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['multimodal_behavioral'], styles)
        story.append(PageBreak())

    # Audio/Voice Analysis
    if analyses.get('audio_voice_analysis'):
        story.append(Paragraph(
            "AUDIO & VOICE ANALYSIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['audio_voice_analysis'], styles)
        story.append(PageBreak())

    # LIWC Linguistic Analysis
    if analyses.get('liwc_linguistic_analysis'):
        story.append(Paragraph(
            "LIWC LINGUISTIC ANALYSIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['liwc_linguistic_analysis'], styles)
        story.append(PageBreak())

    # Personality Synthesis (Big Five, Dark Triad, MBTI)
    if analyses.get('personality_synthesis'):
        story.append(Paragraph(
            "PERSONALITY SYNTHESIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['personality_synthesis'], styles)
        story.append(PageBreak())

    # Threat Assessment Synthesis
    if analyses.get('threat_synthesis'):
        story.append(Paragraph(
            "THREAT ASSESSMENT",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['threat_synthesis'], styles)
        story.append(PageBreak())

    # Differential Diagnosis
    if analyses.get('differential'):
        story.append(Paragraph(
            "DIFFERENTIAL DIAGNOSIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['differential'], styles)

    # Contradictions Analysis
    if analyses.get('contradictions'):
        story.append(Paragraph(
            "CONTRADICTIONS ANALYSIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['contradictions'], styles)

    # Red Team Analysis
    if analyses.get('red_team'):
        story.append(Paragraph(
            "RED TEAM ANALYSIS (SELF-CRITIQUE)",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        _add_analysis_content(story, analyses['red_team'], styles)
        story.append(PageBreak())

    # NCI/Chase Hughes Deception Analysis
    # Collect all NCI-related analyses from the result
    nci_sections = []

    # Check for NCI-specific keys in analyses
    nci_keys = [
        ('blink_rate', 'BLINK RATE ANALYSIS'),
        ('bte_scoring', 'BEHAVIORAL TABLE OF ELEMENTS (BTE)'),
        ('facial_etching', 'FACIAL ETCHING ANALYSIS'),
        ('gestural_mismatch', 'GESTURAL MISMATCH DETECTION'),
        ('stress_clusters', 'STRESS CLUSTER ANALYSIS'),
        ('five_cs', 'FIVE C\'S FRAMEWORK'),
        ('baseline_deviation', 'BASELINE DEVIATION ANALYSIS'),
        ('detail_mountain_valley', 'DETAIL MOUNTAIN/VALLEY ANALYSIS'),
        ('minimizing_language', 'MINIMIZING LANGUAGE ANALYSIS'),
        ('linguistic_harvesting', 'LINGUISTIC HARVESTING'),
        ('fate_model', 'FATE MODEL PROFILE'),
        ('nci_deception_summary', 'NCI DECEPTION SUMMARY'),
    ]

    for key, title in nci_keys:
        if analyses.get(key):
            nci_sections.append((title, analyses[key]))

    # Also check if NCI content is embedded in combined analyses
    # by looking for NCI markers in the text
    for key, content in analyses.items():
        if isinstance(content, str):
            if any(marker in content.upper() for marker in ['BTE SCORE', 'BLINK RATE', 'FIVE C', 'FATE MODEL', 'CHASE HUGHES', 'NCI']):
                # Content has NCI markers - already included in other sections
                pass

    if nci_sections:
        story.append(Paragraph(
            "NCI/CHASE HUGHES DECEPTION ANALYSIS",
            styles['FBISection']
        ))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
            spaceAfter=10
        ))
        story.append(Paragraph(
            "Based on methodologies from The Behavior Ops Manual and Six-Minute X-Ray by Chase Hughes / NCI University",
            styles['FBIMeta']
        ))
        story.append(Spacer(1, 10))

        for title, content in nci_sections:
            story.append(Paragraph(title, styles['FBISubsection']))
            _add_analysis_content(story, content, styles)
            story.append(Spacer(1, 10))

    # Footer
    story.append(Spacer(1, 30))
    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=FBI_ACCENT,
        spaceBefore=20,
        spaceAfter=10
    ))
    story.append(Paragraph(
        "This report was generated by the FBI-Style Behavioral Profiling System. "
        "For research and educational purposes only. Subject consent required.",
        styles['FBIMeta']
    ))

    # Build PDF
    doc.build(story)

    logger.info(f"PDF report generated: {pdf_path}")
    return str(pdf_path)


def _add_analysis_content(story, content: str, styles):
    """
    Add analysis content to the PDF story, parsing sections.

    Args:
        story: PDF story list to append to
        content: Raw analysis text content
        styles: PDF styles dictionary
    """
    if not content:
        story.append(Paragraph("No analysis available.", styles['FBIBody']))
        return

    lines = content.strip().split('\n')
    current_text = []

    for line in lines:
        line = line.strip()

        if not line:
            # Empty line - flush current text
            if current_text:
                story.append(Paragraph(' '.join(current_text), styles['FBIBody']))
                current_text = []
            story.append(Spacer(1, 6))
            continue

        # Check for section headers (lines starting with ## or **)
        if line.startswith('##'):
            # Flush current text
            if current_text:
                story.append(Paragraph(' '.join(current_text), styles['FBIBody']))
                current_text = []
            # Add as subsection
            header_text = line.lstrip('#').strip()
            story.append(Paragraph(header_text, styles['FBISubsection']))
            continue

        if line.startswith('**') and line.endswith('**'):
            # Bold header
            if current_text:
                story.append(Paragraph(' '.join(current_text), styles['FBIBody']))
                current_text = []
            header_text = line.strip('*').strip()
            story.append(Paragraph(f"<b>{header_text}</b>", styles['FBIBody']))
            continue

        if line.startswith('- ') or line.startswith('* '):
            # Bullet point
            if current_text:
                story.append(Paragraph(' '.join(current_text), styles['FBIBody']))
                current_text = []
            bullet_text = line[2:].strip()
            # Convert markdown bold to reportlab bold
            bullet_text = bullet_text.replace('**', '')
            story.append(Paragraph(f"• {bullet_text}", styles['FBIBody']))
            continue

        # Regular text - accumulate
        # Clean up markdown formatting
        line = line.replace('**', '')
        line = line.replace('*', '')
        current_text.append(line)

    # Flush remaining text
    if current_text:
        story.append(Paragraph(' '.join(current_text), styles['FBIBody']))


def generate_summary_pdf(profiles: list, subject_name: str, output_path: str = None) -> str:
    """
    Generate a summary PDF for multiple profiles of the same subject.

    Args:
        profiles: List of profile result dictionaries
        subject_name: Name of the subject
        output_path: Optional output path

    Returns:
        Path to generated PDF
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF generation")

    if output_path:
        pdf_path = Path(output_path)
    else:
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.pdf',
            prefix=f'profile_summary_{subject_name}_'
        )
        pdf_path = Path(temp_file.name)
        temp_file.close()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = create_pdf_styles()
    story = []

    # Header
    story.append(Paragraph(
        "CONFIDENTIAL - FOR RESEARCH USE ONLY",
        styles['FBIClassification']
    ))
    story.append(Paragraph(
        f"PROFILE SUMMARY: {subject_name.upper()}",
        styles['FBITitle']
    ))

    # Summary statistics
    story.append(Paragraph(
        f"Total Analyses: {len(profiles)}",
        styles['FBIBody']
    ))

    if profiles:
        first_date = profiles[-1].get('timestamp', 'N/A')[:10]
        last_date = profiles[0].get('timestamp', 'N/A')[:10]
        story.append(Paragraph(
            f"Analysis Period: {first_date} to {last_date}",
            styles['FBIBody']
        ))

    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=FBI_ACCENT,
        spaceBefore=15,
        spaceAfter=15
    ))

    # Profile timeline
    story.append(Paragraph("ANALYSIS TIMELINE", styles['FBISection']))

    for i, profile in enumerate(profiles):
        report_num = len(profiles) - i
        timestamp = profile.get('timestamp', 'N/A')[:19]
        case_id = profile.get('case_id', 'N/A')

        story.append(Paragraph(
            f"<b>Report #{report_num}</b> - {timestamp}",
            styles['FBIBody']
        ))
        story.append(Paragraph(
            f"Case ID: {case_id}",
            styles['FBIMeta']
        ))
        story.append(Spacer(1, 10))

    doc.build(story)
    logger.info(f"Summary PDF generated: {pdf_path}")
    return str(pdf_path)
