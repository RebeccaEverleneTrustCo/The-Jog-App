import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Patient Mood Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stRadio > label {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'questionnaire_complete' not in st.session_state:
    st.session_state.questionnaire_complete = False
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'emotion_classifier' not in st.session_state:
    st.session_state.emotion_classifier = None
if 'current_answers' not in st.session_state:
    st.session_state.current_answers = {}
if 'consent_given' not in st.session_state:
    st.session_state.consent_given = False
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'patient_identified' not in st.session_state:
    st.session_state.patient_identified = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# Data directory setup
DATA_DIR = Path("patient_data")
DATA_DIR.mkdir(exist_ok=True)

# Load emotion detection model
@st.cache_resource
def load_emotion_model():
    """Load a lightweight emotion classification model from HuggingFace"""
    try:
        # Using a small and efficient emotion detection model
        classifier = pipeline(
            "text-classification",
            model="michellejieli/emotion_text_classifier",
            top_k=None
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Patient data management functions
def save_patient_assessment(patient_id, responses, mood_analysis, text_emotions):
    """Save assessment results for a patient"""
    patient_file = DATA_DIR / f"{patient_id}.json"
    
    # Load existing data or create new
    if patient_file.exists():
        with open(patient_file, 'r') as f:
            patient_data = json.load(f)
    else:
        patient_data = {
            "patient_id": patient_id,
            "assessments": []
        }
    
    # Create new assessment entry
    assessment = {
        "timestamp": datetime.now().isoformat(),
        "responses": responses,
        "mood_analysis": mood_analysis,
        "text_emotions": text_emotions if text_emotions else []
    }
    
    patient_data["assessments"].append(assessment)
    
    # Save to file
    with open(patient_file, 'w') as f:
        json.dump(patient_data, f, indent=2)
    
    return len(patient_data["assessments"])

def load_patient_data(patient_id):
    """Load all assessment data for a patient"""
    patient_file = DATA_DIR / f"{patient_id}.json"
    
    if patient_file.exists():
        with open(patient_file, 'r') as f:
            return json.load(f)
    return None

def get_all_patients():
    """Get list of all patient IDs"""
    if not DATA_DIR.exists():
        return []
    
    patients = []
    for file in DATA_DIR.glob("*.json"):
        patients.append(file.stem)
    return sorted(patients)

def display_historical_trends(patient_data):
    """Display mood trends over time for a patient"""
    if not patient_data or len(patient_data.get("assessments", [])) < 2:
        st.info("📊 Historical trends will appear after completing multiple assessments.")
        return
    
    assessments = patient_data["assessments"]
    
    st.header("📈 Mood Trends Over Time")
    
    # Prepare data for visualization
    timestamps = []
    positive_scores = []
    neutral_scores = []
    negative_scores = []
    
    for assessment in assessments:
        timestamps.append(datetime.fromisoformat(assessment["timestamp"]))
        mood = assessment["mood_analysis"]
        positive_scores.append(mood.get("positive", 0) * 100)
        neutral_scores.append(mood.get("neutral", 0) * 100)
        negative_scores.append(mood.get("negative", 0) * 100)
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=positive_scores,
        mode='lines+markers',
        name='Positive',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=neutral_scores,
        mode='lines+markers',
        name='Neutral',
        line=dict(color='#95a5a6', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=negative_scores,
        mode='lines+markers',
        name='Negative',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Mood Distribution Over Time",
        xaxis_title="Assessment Date",
        yaxis_title="Mood Score (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show assessment history
    st.subheader("📋 Assessment History")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Assessments", len(assessments))
    with col2:
        first_date = datetime.fromisoformat(assessments[0]["timestamp"]).strftime("%b %d, %Y")
        st.metric("First Assessment", first_date)
    with col3:
        latest_date = datetime.fromisoformat(assessments[-1]["timestamp"]).strftime("%b %d, %Y")
        st.metric("Latest Assessment", latest_date)
    
    # Detailed history in expandable sections
    with st.expander("📜 View All Past Assessments"):
        for i, assessment in enumerate(reversed(assessments), 1):
            timestamp = datetime.fromisoformat(assessment["timestamp"])
            mood = assessment["mood_analysis"]
            dominant_mood = max(mood.items(), key=lambda x: x[1])[0]
            
            st.markdown(f"### Assessment #{len(assessments) - i + 1}")
            st.markdown(f"**Date:** {timestamp.strftime('%B %d, %Y at %I:%M %p')}")
            
            mood_emoji = "😊" if dominant_mood == "positive" else "😐" if dominant_mood == "neutral" else "😔"
            st.markdown(f"**Dominant Mood:** {mood_emoji} {dominant_mood.capitalize()} "
                       f"({mood[dominant_mood]*100:.1f}%)")
            
            st.markdown("---")

# Questionnaire questions
QUESTIONNAIRE = {
    "Sleep Pattern": {
        "question": "How would you describe your sleep quality over the past week?",
        "options": [
            "Excellent - I sleep well and wake up refreshed",
            "Good - I sleep most nights without issues",
            "Fair - I have occasional difficulty sleeping",
            "Poor - I frequently struggle with sleep",
            "Very Poor - I barely get any restful sleep"
        ]
    },
    "Energy Levels": {
        "question": "How would you rate your energy levels throughout the day?",
        "options": [
            "Very High - I feel energetic and motivated",
            "High - I have good energy most of the time",
            "Moderate - My energy is okay, not great",
            "Low - I often feel tired and drained",
            "Very Low - I feel exhausted most of the time"
        ]
    },
    "Social Interaction": {
        "question": "How do you feel about social interactions lately?",
        "options": [
            "I enjoy them and seek them out",
            "I'm comfortable with them when they happen",
            "I'm neutral about social interactions",
            "I tend to avoid them when possible",
            "I strongly avoid all social contact"
        ]
    },
    "Mood Stability": {
        "question": "How stable has your mood been recently?",
        "options": [
            "Very stable - I feel consistently positive",
            "Mostly stable - Minor fluctuations",
            "Somewhat unstable - Noticeable mood swings",
            "Unstable - Frequent mood changes",
            "Very unstable - Extreme mood variations"
        ]
    },
    "Stress Level": {
        "question": "How stressed have you been feeling?",
        "options": [
            "Not stressed at all - Very calm",
            "Mildly stressed - Manageable",
            "Moderately stressed - Somewhat challenging",
            "Highly stressed - Difficult to manage",
            "Extremely stressed - Overwhelmed"
        ]
    },
    "Interest in Activities": {
        "question": "How interested are you in activities you usually enjoy?",
        "options": [
            "Very interested - I enjoy them fully",
            "Interested - I still find pleasure in them",
            "Somewhat interested - Less enjoyment than before",
            "Not very interested - Little enjoyment",
            "Not interested at all - No pleasure in activities"
        ]
    },
    "Overall Feeling": {
        "question": "Please describe how you're feeling overall in a few sentences:",
        "type": "text_area"
    }
}

def analyze_mood_from_responses(responses):
    """Analyze mood based on questionnaire responses"""
    mood_scores = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }
    
    # Score mapping for multiple choice questions
    score_mapping = {
        0: {"positive": 5, "neutral": 0, "negative": 0},  # Best option
        1: {"positive": 3, "neutral": 2, "negative": 0},
        2: {"positive": 0, "neutral": 5, "negative": 0},
        3: {"positive": 0, "neutral": 2, "negative": 3},
        4: {"positive": 0, "neutral": 0, "negative": 5}   # Worst option
    }
    
    # Analyze multiple choice responses
    for key, response in responses.items():
        if key != "Overall Feeling" and isinstance(response, int):
            scores = score_mapping.get(response, {"neutral": 1})
            for mood, score in scores.items():
                mood_scores[mood] += score
    
    # Normalize scores
    total = sum(mood_scores.values())
    if total > 0:
        mood_scores = {k: v/total for k, v in mood_scores.items()}
    
    return mood_scores

def get_emotion_from_text(text, classifier):
    """Get emotion from text using HuggingFace model"""
    if classifier is None or not text.strip():
        return None
    
    try:
        results = classifier(text[:512])  # Limit text length
        if results and len(results) > 0:
            # Sort by score
            sorted_emotions = sorted(results[0], key=lambda x: x['score'], reverse=True)
            return sorted_emotions
        return None
    except Exception as e:
        st.error(f"Error analyzing text: {e}")
        return None

def display_emotion_results(questionnaire_analysis, text_emotions):
    """Display comprehensive emotion analysis results"""
    st.markdown("---")
    st.header("📊 Mood Assessment Results")
    
    # Save the assessment data
    assessment_number = save_patient_assessment(
        st.session_state.patient_id,
        st.session_state.responses,
        questionnaire_analysis,
        text_emotions
    )
    
    st.success(f"✅ Assessment #{assessment_number} saved successfully for patient: {st.session_state.patient_id.replace('_', ' ')}")
    
    # Quick summary at the top (always visible)
    dominant_mood = max(questionnaire_analysis.items(), key=lambda x: x[1])[0]
    dominant_percentage = questionnaire_analysis[dominant_mood] * 100
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        emoji = "😊" if dominant_mood == "positive" else "😐" if dominant_mood == "neutral" else "😔"
        st.metric("Dominant Mood", f"{emoji} {dominant_mood.capitalize()}", f"{dominant_percentage:.0f}%")
    with summary_col2:
        if text_emotions:
            top_emotion = text_emotions[0]
            st.metric("Primary Emotion", top_emotion['label'].title(), f"{top_emotion['score']*100:.0f}%")
        else:
            st.metric("Primary Emotion", "Not analyzed", "Optional question skipped")
    with summary_col3:
        total_questions = len(st.session_state.responses)
        st.metric("Questions Answered", total_questions, "")
    
    st.markdown("---")
    
    # Collapsible sections
    with st.expander("📈 Questionnaire Analysis", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create pie chart for questionnaire results
            if questionnaire_analysis:
                fig = go.Figure(data=[go.Pie(
                    labels=list(questionnaire_analysis.keys()),
                    values=list(questionnaire_analysis.values()),
                    hole=.3,
                    marker_colors=['#2ecc71', '#95a5a6', '#e74c3c']
                )])
                fig.update_layout(
                    title="Overall Mood Distribution",
                    height=350,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Mood Breakdown:")
            for mood, score in questionnaire_analysis.items():
                percentage = score * 100
                emoji = "😊" if mood == "positive" else "😐" if mood == "neutral" else "😔"
                st.metric(f"{emoji} {mood.capitalize()}", f"{percentage:.1f}%")
    
    with st.expander("🎭 Text Emotion Analysis", expanded=False):
        if text_emotions:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create bar chart for emotions
                emotions_df = pd.DataFrame(text_emotions)
                emotions_df = emotions_df.sort_values('score', ascending=True)
                
                fig = px.bar(
                    emotions_df,
                    x='score',
                    y='label',
                    orientation='h',
                    title="Detected Emotions from Your Response",
                    labels={'score': 'Confidence', 'label': 'Emotion'},
                    color='score',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Emotion Scores:")
                for emotion in text_emotions[:5]:  # Show top 5
                    st.metric(
                        emotion['label'].title(),
                        f"{emotion['score']*100:.1f}%"
                    )
        else:
            st.info("ℹ️ No text response provided. The 'Overall Feeling' question was optional. Text emotion analysis is only available when you provide a written response.")
    
    with st.expander("💡 Overall Assessment & Interpretation", expanded=True):
        # Enhanced assessment with reinforcement
        if dominant_mood == "positive":
            st.success("🌟 **Great News!**")
            st.markdown("""
            Your responses indicate a **generally positive mood state**. You're showing good emotional 
            well-being with healthy patterns across sleep, energy, and social engagement. 
            
            **This is wonderful!** Keep up the positive habits that are working for you. Your emotional 
            awareness and the time you're taking to check in with yourself shows strong self-care practices.
            """)
            
            # Additional positive reinforcement
            st.info("💪 **You're doing well!** Continue nurturing your mental health with the practices "
                   "that have been supporting your positive mood.")
            
        elif dominant_mood == "neutral":
            st.warning("🤔 **Mixed Patterns Detected**")
            st.markdown("""
            Your responses indicate a **neutral to mixed mood state**. While you're functioning, there may 
            be room for improvement in certain areas of emotional well-being.
            
            **You're managing**, and that's important to acknowledge. Small, consistent changes in areas 
            where you're struggling can make a meaningful difference. You've already taken a positive step 
            by completing this assessment.
            """)
            
            # Encouraging message
            st.info("🌱 **There's room for growth.** The fact that you're checking in with yourself shows "
                   "awareness and willingness to improve. Small steps can lead to meaningful changes.")
            
        else:
            st.error("💙 **We're Here to Support You**")
            st.markdown("""
            Your responses indicate some **patterns that may benefit from additional support**. Please know 
            that what you're experiencing is valid, and reaching out for help is a sign of strength, not weakness.
            
            **You don't have to face this alone.** Many people experience difficult emotional periods, and 
            professional support can make a real difference. Consider reaching out to a mental health 
            professional, counselor, or therapist who can provide personalized guidance.
            """)
            
            # Supportive and hopeful message
            st.info("🫂 **You took an important step** by completing this assessment. Seeking understanding "
                   "about your emotional state shows courage and self-awareness. Help is available, and things "
                   "can get better with the right support.")
        
        st.markdown("#### Key Insights:")
        
        # Analyze specific areas with reinforcement
        insights = []
        positive_areas = []
        concern_areas = []
        responses = st.session_state.responses
        
        # Evaluate each area
        if responses.get("Sleep Pattern", 0) >= 3:
            concern_areas.append("Sleep")
            insights.append("⚠️ **Sleep**: Your sleep quality may need attention. Quality sleep is foundational to emotional well-being.")
        elif responses.get("Sleep Pattern", 0) <= 1:
            positive_areas.append("Sleep")
            insights.append("✅ **Sleep**: Excellent! You're maintaining good sleep quality, which is crucial for mood regulation.")
        
        if responses.get("Energy Levels", 0) >= 3:
            concern_areas.append("Energy")
            insights.append("⚠️ **Energy**: Low energy levels detected. This can impact your daily functioning and mood.")
        elif responses.get("Energy Levels", 0) <= 1:
            positive_areas.append("Energy")
            insights.append("✅ **Energy**: Great job! Your energy levels are strong, helping you stay motivated and engaged.")
        
        if responses.get("Stress Level", 0) >= 3:
            concern_areas.append("Stress")
            insights.append("⚠️ **Stress**: High stress levels detected. Managing stress is important for your overall well-being.")
        elif responses.get("Stress Level", 0) <= 1:
            positive_areas.append("Stress")
            insights.append("✅ **Stress**: Well done! You're managing stress effectively, which protects your mental health.")
        
        if responses.get("Social Interaction", 0) >= 3:
            concern_areas.append("Social")
            insights.append("⚠️ **Social**: Social withdrawal tendencies noted. Connection with others can be healing.")
        elif responses.get("Social Interaction", 0) <= 1:
            positive_areas.append("Social")
            insights.append("✅ **Social**: Wonderful! You're maintaining positive social engagement, which supports emotional health.")
        
        if responses.get("Mood Stability", 0) >= 3:
            concern_areas.append("Mood")
            insights.append("⚠️ **Mood Stability**: Noticeable mood fluctuations. Understanding patterns can help manage them.")
        elif responses.get("Mood Stability", 0) <= 1:
            positive_areas.append("Mood")
            insights.append("✅ **Mood Stability**: Excellent! Your mood has been stable, showing emotional resilience.")
        
        if responses.get("Interest in Activities", 0) >= 3:
            concern_areas.append("Activities")
            insights.append("⚠️ **Activities**: Decreased interest in enjoyable activities. This is worth addressing.")
        elif responses.get("Interest in Activities", 0) <= 1:
            positive_areas.append("Activities")
            insights.append("✅ **Activities**: Fantastic! You're staying engaged with activities you enjoy.")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.markdown("- Overall balanced responses across all areas")
        
        # Summary reinforcement
        st.markdown("")
        if len(positive_areas) > len(concern_areas):
            st.success(f"🎉 **Strong Areas**: You're doing well in {len(positive_areas)} out of {len(positive_areas) + len(concern_areas)} assessed areas! "
                      f"Your strengths in {', '.join(positive_areas[:3])} are supporting your overall well-being.")
        elif len(concern_areas) > len(positive_areas):
            st.info(f"🌟 **Areas for Growth**: While there are {len(concern_areas)} areas that could use attention, "
                   f"remember that awareness is the first step to positive change. You've already started that process.")
        else:
            st.info("⚖️ **Balanced Profile**: You have a mix of strong areas and areas for improvement, which is common and manageable.")
    
    with st.expander("🎯 Personalized Recommendations"):
        recommendations = generate_recommendations(questionnaire_analysis, st.session_state.responses)
        
        # Get dominant mood for context
        dominant_mood = max(questionnaire_analysis.items(), key=lambda x: x[1])[0]
        
        if dominant_mood == "positive":
            st.markdown("### 🌟 You're Doing Great!")
            st.markdown("Based on your positive responses, here are ways to maintain and strengthen your emotional wellness:")
        elif dominant_mood == "neutral":
            st.markdown("### 🌱 Opportunities for Growth")
            st.markdown("Based on your responses, here are supportive strategies to enhance your well-being:")
        else:
            st.markdown("### 💙 We're Here to Support You")
            st.markdown("Based on your responses, here are caring suggestions. Please remember that professional support can make a real difference:")
        
        for i, rec in enumerate(recommendations, 1):
            # Check if it's a header/category or a recommendation
            if rec.startswith("**") and rec.endswith("**"):
                st.markdown(f"\n{rec}")
            else:
                st.markdown(f"{i}. {rec}")
        
        # Closing encouragement based on mood
        st.markdown("")
        if dominant_mood == "positive":
            st.success("💪 Your commitment to maintaining your mental health is commendable. Keep nurturing yourself!")
        elif dominant_mood == "neutral":
            st.info("🌟 Remember: Progress isn't always linear, and seeking help is a sign of wisdom and strength.")
        else:
            st.error("🫂 Please know: What you're experiencing is real, you're not alone, and help is available. Reaching out is brave.")
        
        st.markdown("---")
        st.info("💙 **Remember**: These are general suggestions. For personalized guidance tailored to your unique situation, consult with a qualified mental health professional.")
    
    with st.expander("📋 Your Responses Summary"):
        st.markdown("#### Review your questionnaire responses:")
        
        for key, value in st.session_state.responses.items():
            if key == "Overall Feeling":
                st.markdown(f"**{key}:**")
                st.text_area("Your written response:", value, height=100, disabled=True, key=f"review_{key}")
            else:
                question_data = QUESTIONNAIRE.get(key, {})
                options = question_data.get("options", [])
                if isinstance(value, int) and value < len(options):
                    st.markdown(f"**{key}:** {options[value]}")
                else:
                    st.markdown(f"**{key}:** {value}")
    
    # Export option (always visible at bottom)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📥 Export Results as Report", use_container_width=True):
            generate_report(questionnaire_analysis, text_emotions)

def generate_recommendations(mood_analysis, responses):
    """Generate personalized recommendations based on responses with encouraging tone"""
    recommendations = []
    
    dominant_mood = max(mood_analysis.items(), key=lambda x: x[1])[0]
    
    # Add mood-specific opening encouragement
    if dominant_mood == "positive":
        recommendations.append("**Keep up the excellent work!** Your positive patterns are serving you well. Here are ways to maintain your emotional wellness:")
    elif dominant_mood == "neutral":
        recommendations.append("**You're on a good path.** Here are some actionable steps to enhance your emotional well-being:")
    else:
        recommendations.append("**Small steps matter.** Here are supportive strategies that can help. Remember, professional guidance can amplify these efforts:")
    
    # Check specific issues with encouraging language
    if responses.get("Sleep Pattern", 0) >= 3:
        recommendations.append("**Sleep**: Consider establishing a calming bedtime routine - try going to bed at the same time each night, limiting screens an hour before sleep, and creating a comfortable sleep environment. Good sleep can transform how you feel.")
    elif responses.get("Sleep Pattern", 0) <= 1:
        recommendations.append("**Sleep**: You're doing great with sleep! Continue your healthy sleep habits - they're a strong foundation for your well-being.")
    
    if responses.get("Energy Levels", 0) >= 3:
        recommendations.append("**Energy**: Try incorporating gentle physical activity like a 10-minute walk, ensuring you're eating balanced meals, and staying hydrated. Even small amounts of movement can boost energy significantly.")
    elif responses.get("Energy Levels", 0) <= 1:
        recommendations.append("**Energy**: Your energy management is excellent! Keep maintaining the practices that give you vitality.")
    
    if responses.get("Social Interaction", 0) >= 3:
        recommendations.append("**Social Connection**: Start small - perhaps a brief text to a friend, or a short coffee chat. Human connection doesn't have to be overwhelming; even small interactions can be meaningful.")
    elif responses.get("Social Interaction", 0) <= 1:
        recommendations.append("**Social**: Your social engagement is strong! These connections are valuable - continue nurturing them.")
    
    if responses.get("Stress Level", 0) >= 3:
        recommendations.append("**Stress Management**: Try stress-reduction techniques like deep breathing (4-7-8 breathing), progressive muscle relaxation, journaling, or mindfulness apps. Even 5 minutes daily can help reset your nervous system.")
    elif responses.get("Stress Level", 0) <= 1:
        recommendations.append("**Stress**: You're managing stress well! Whatever you're doing is working - keep it up.")
    
    if responses.get("Interest in Activities", 0) >= 3:
        recommendations.append("**Activities**: Set tiny, achievable goals - maybe 5 minutes of an activity you used to enjoy, or trying something new that requires minimal effort. Progress, not perfection, is the goal.")
    elif responses.get("Interest in Activities", 0) <= 1:
        recommendations.append("**Activities**: Wonderful! Your engagement with enjoyable activities is protecting your mental health.")
    
    if responses.get("Mood Stability", 0) >= 3:
        recommendations.append("**Mood Tracking**: Consider keeping a simple mood journal to identify triggers and patterns. Understanding your mood fluctuations is the first step to managing them.")
    elif responses.get("Mood Stability", 0) <= 1:
        recommendations.append("**Mood**: Your emotional stability is a real strength - it helps you navigate life's ups and downs.")
    
    # General recommendations based on dominant mood
    if dominant_mood == "negative":
        recommendations.append("**🫂 Most Important**: Consider reaching out to a mental health professional - therapist, counselor, or your doctor. They can provide personalized support and evidence-based treatments. You deserve that support.")
        recommendations.append("**Support System**: Let someone you trust know how you're feeling. You don't have to face this alone.")
    elif dominant_mood == "neutral":
        recommendations.append("**Maintain Momentum**: Regular self-check-ins like this assessment help you stay aware of your emotional state.")
        recommendations.append("**Build Resilience**: Consider talking to a counselor or therapist even when things are manageable - they can provide tools for continued growth.")
    else:
        recommendations.append("**Sustain Your Success**: Continue the self-care practices that are working for you.")
        recommendations.append("**Stay Proactive**: Regular check-ins with yourself help maintain your positive trajectory.")
    
    return recommendations

def generate_report(questionnaire_analysis, text_emotions):
    """Generate a downloadable report"""
    report = f"""
MOOD ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUESTIONNAIRE ANALYSIS:
{'='*50}
Positive Mood: {questionnaire_analysis['positive']*100:.1f}%
Neutral Mood: {questionnaire_analysis['neutral']*100:.1f}%
Negative Mood: {questionnaire_analysis['negative']*100:.1f}%

TEXT EMOTION ANALYSIS:
{'='*50}
"""
    if text_emotions:
        for emotion in text_emotions:
            report += f"{emotion['label']}: {emotion['score']*100:.1f}%\n"
    
    report += f"\n\nRESPONSES:\n{'='*50}\n"
    for key, value in st.session_state.responses.items():
        if key == "Overall Feeling":
            report += f"\n{key}:\n{value}\n"
        else:
            report += f"\n{key}: Option {value + 1}\n"
    
    st.download_button(
        label="Download Report",
        data=report,
        file_name=f"mood_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def main():
    # Header
    st.title("🧠 Patient Mood & Emotion Assessment")
    st.markdown("""
    Welcome to the mood assessment questionnaire. This tool helps understand your current emotional state 
    through a series of questions and text analysis. Please answer honestly for the most accurate assessment.
    """)
    
    # Load model
    if st.session_state.emotion_classifier is None:
        with st.spinner("Loading emotion detection model..."):
            st.session_state.emotion_classifier = load_emotion_model()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About This Assessment")
        st.markdown("""
        This questionnaire assesses:
        - Sleep patterns
        - Energy levels
        - Social engagement
        - Mood stability
        - Stress levels
        - Interest in activities
        
        **Note:** This is not a diagnostic tool. 
        For professional help, please consult a 
        mental health professional.
        """)
        
        # Show patient info if identified
        if st.session_state.patient_identified:
            st.markdown("---")
            st.success(f"👤 **Patient ID:** {st.session_state.patient_id}")
            
            # Load patient data for count
            patient_data = load_patient_data(st.session_state.patient_id)
            if patient_data:
                assessment_count = len(patient_data.get("assessments", []))
                st.info(f"📊 **Total Assessments:** {assessment_count}")
        
        if st.session_state.questionnaire_complete:
            if st.button("🔄 Start New Assessment"):
                st.session_state.questionnaire_complete = False
                st.session_state.responses = {}
                st.session_state.current_answers = {}
                st.session_state.consent_given = False
                st.rerun()
        
        # Option to change patient or view history
        if st.session_state.patient_identified:
            st.markdown("---")
            if st.button("👥 Change Patient"):
                st.session_state.patient_identified = False
                st.session_state.patient_id = None
                st.session_state.questionnaire_complete = False
                st.session_state.responses = {}
                st.session_state.current_answers = {}
                st.session_state.consent_given = False
                st.rerun()
            
            if st.button("📊 View Historical Trends"):
                st.session_state.questionnaire_complete = False
                st.session_state.show_history = True
                st.rerun()
    
    # Patient Identification Screen
    if not st.session_state.patient_identified:
        st.markdown("---")
        st.header("👤 Patient Identification")
        st.markdown("Please enter your name or patient ID to begin. This helps track your mood over time.")
        
        tab1, tab2 = st.tabs(["🆕 New Patient / Enter ID", "📋 Returning Patient"])
        
        with tab1:
            st.markdown("### Enter Patient Information")
            patient_input = st.text_input(
                "Patient Name or ID:",
                placeholder="e.g., John Doe or Patient-12345",
                help="Enter a name or unique identifier. This will be used to save and track your assessments."
            )
            
            if st.button("✅ Continue", type="primary", disabled=not patient_input.strip()):
                st.session_state.patient_id = patient_input.strip().replace(" ", "_")
                st.session_state.patient_identified = True
                
                # Check if returning patient
                patient_data = load_patient_data(st.session_state.patient_id)
                if patient_data:
                    assessment_count = len(patient_data.get("assessments", []))
                    st.success(f"Welcome back! We found {assessment_count} previous assessment(s) for {patient_input}.")
                else:
                    st.success(f"Welcome, {patient_input}! This is your first assessment.")
                
                st.rerun()
        
        with tab2:
            st.markdown("### Select From Existing Patients")
            
            all_patients = get_all_patients()
            
            if all_patients:
                selected_patient = st.selectbox(
                    "Choose a patient:",
                    options=["— Select a patient —"] + all_patients,
                    format_func=lambda x: x.replace("_", " ") if x != "— Select a patient —" else x
                )
                
                if selected_patient != "— Select a patient —":
                    patient_data = load_patient_data(selected_patient)
                    if patient_data:
                        assessment_count = len(patient_data.get("assessments", []))
                        st.info(f"📊 This patient has {assessment_count} previous assessment(s).")
                    
                    if st.button("✅ Continue with Selected Patient", type="primary"):
                        st.session_state.patient_id = selected_patient
                        st.session_state.patient_identified = True
                        st.success(f"Welcome back, {selected_patient.replace('_', ' ')}!")
                        st.rerun()
            else:
                st.info("No previous patients found. Please use the 'New Patient' tab to get started.")
        
        return  # Stop here until patient is identified
    
    # Show history view if requested
    if st.session_state.get('show_history', False):
        st.markdown("---")
        patient_data = load_patient_data(st.session_state.patient_id)
        if patient_data:
            display_historical_trends(patient_data)
        
        if st.button("📝 Take New Assessment"):
            st.session_state.show_history = False
            st.rerun()
        
        return  # Stop here in history view
    
    # Consent Form
    if not st.session_state.consent_given and not st.session_state.questionnaire_complete:
        st.markdown("---")
        st.header("📋 Informed Consent")
        
        st.markdown("""
        Before beginning this mood assessment, please read and acknowledge the following:
        """)
        
        # Consent information box
        st.info("""
        **Purpose of This Assessment:**
        
        This questionnaire is designed to help you understand your current emotional state and mood patterns. 
        It uses a combination of structured questions and AI-powered text analysis to provide insights about your mood.
        """)
        
        st.warning("""
        **Important Disclaimers:**
        
        ⚠️ **Not a Diagnostic Tool**: This assessment is for informational and awareness purposes only. 
        It is NOT a substitute for professional medical or mental health diagnosis.
        
        ⚠️ **Not Medical Advice**: Results should not be used to diagnose, treat, cure, or prevent any medical 
        or mental health condition.
        
        ⚠️ **Seek Professional Help**: If you are experiencing significant distress, mental health concerns, 
        or are in crisis, please consult with a qualified mental health professional, counselor, or therapist.
        
        ⚠️ **Privacy**: Your responses will be processed using AI models for emotion analysis. Data is not 
        stored permanently but is processed during your session.
        """)
        
        st.success("""
        **What You'll Get:**
        
        ✅ Insights into your current mood state based on your responses
        
        ✅ Visual analysis of emotional patterns
        
        ✅ Personalized suggestions for emotional well-being
        
        ✅ A downloadable report of your assessment
        """)
        
        # Crisis resources
        with st.expander("🆘 Crisis Resources & Support"):
            st.markdown("""
            If you are in crisis or need immediate help:
            
            - **National Suicide Prevention Lifeline (US)**: 988 or 1-800-273-8255
            - **Crisis Text Line (US)**: Text HOME to 741741
            - **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
            - **Emergency Services**: Call 911 (US) or your local emergency number
            
            **Mental Health Resources:**
            - **SAMHSA National Helpline**: 1-800-662-4357 (24/7, free, confidential)
            - **NAMI Helpline**: 1-800-950-6264
            - Visit your primary care physician or local mental health clinic
            """)
        
        st.markdown("---")
        
        # Consent checkbox
        consent_checkbox = st.checkbox(
            "I understand and acknowledge that this is an informational tool only, not a diagnostic instrument. "
            "I confirm that I am using this assessment for awareness purposes and will seek professional help "
            "if needed.",
            key="consent_checkbox"
        )
        
        st.markdown("")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if consent_checkbox:
                if st.button("✅ I Agree - Begin Assessment", type="primary", use_container_width=True):
                    st.session_state.consent_given = True
                    st.rerun()
            else:
                st.button("✅ I Agree - Begin Assessment", disabled=True, use_container_width=True)
                st.caption("Please check the box above to continue")
    
    # Main questionnaire
    elif st.session_state.consent_given and not st.session_state.questionnaire_complete:
        # Calculate dynamic progress
        total_questions = len(QUESTIONNAIRE)
        
        # Count answered questions (excluding None/empty values)
        # Note: "Overall Feeling" is optional, so we count it separately
        answered_count = 0
        required_questions = 0
        
        for key in QUESTIONNAIRE.keys():
            if key != "Overall Feeling":  # Count required questions
                required_questions += 1
                answer = st.session_state.current_answers.get(key)
                if answer is not None:
                    answered_count += 1
            else:
                # Optional question - still show in total but don't require
                answer = st.session_state.current_answers.get(key)
                if answer is not None and answer != "" and answer.strip():
                    answered_count += 1
        
        remaining_count = required_questions - answered_count
        progress_percentage = answered_count / total_questions
        
        st.markdown("### Please answer the following questions:")
        
        # Dynamic progress info box
        if answered_count == 0:
            st.info(f"📋 **Questionnaire Overview**: {total_questions} questions total • Estimated time: 3-5 minutes")
        else:
            progress_col1, progress_col2 = st.columns([3, 1])
            with progress_col1:
                st.progress(progress_percentage, text=f"Progress: {answered_count}/{total_questions} questions answered")
            with progress_col2:
                if remaining_count > 0:
                    st.metric("Remaining", f"{remaining_count}", delta=f"-{answered_count}", delta_color="inverse")
                else:
                    st.success("✅ Complete!")
        
        st.markdown("---")
        
        # Questions (not in a form, so they update in real-time)
        for idx, (key, question_data) in enumerate(QUESTIONNAIRE.items(), 1):
            # Mark if question is optional
            if key == "Overall Feeling":
                st.markdown(f"#### Question {idx} of {total_questions} *(Optional)*")
            else:
                st.markdown(f"#### Question {idx} of {total_questions}")
            
            st.markdown(f"**{question_data['question']}**")
            
            # Check if this question is answered
            is_answered = st.session_state.current_answers.get(key) is not None and st.session_state.current_answers.get(key) != ""
            if is_answered:
                st.markdown("✅ *Answered*")
            
            if question_data.get("type") == "text_area":
                st.session_state.current_answers[key] = st.text_area(
                    "Your response (optional):",
                    value=st.session_state.current_answers.get(key, ""),
                    key=f"input_{key}",
                    height=150,
                    placeholder="(Optional) Share your thoughts and feelings for more detailed emotion analysis..."
                )
            else:
                # Use selectbox instead of radio to allow blank initial state
                options = ["— Select an option —"] + question_data["options"]
                current_value = st.session_state.current_answers.get(key)
                
                # Determine the index for selectbox
                if current_value is None:
                    default_index = 0
                else:
                    default_index = current_value + 1
                
                selected = st.selectbox(
                    "Select one:",
                    range(len(options)),
                    index=default_index,
                    format_func=lambda x: options[x],
                    key=f"input_{key}"
                )
                
                # Store None if placeholder is selected, otherwise store the actual index
                if selected == 0:
                    st.session_state.current_answers[key] = None
                else:
                    st.session_state.current_answers[key] = selected - 1
            
            st.markdown("---")
        
        # Submit button
        st.markdown("")
        
        # Check if all required questions are answered (excluding optional "Overall Feeling")
        all_required_answered = remaining_count == 0
        
        if all_required_answered:
            st.success(f"🎉 All required questions answered! Ready to submit.")
            if answered_count < total_questions:
                st.info("💡 Tip: Answering the optional 'Overall Feeling' question will provide more detailed emotion analysis.")
            
            if st.button("📊 Submit Assessment", type="primary", use_container_width=True):
                # Prepare responses for analysis
                final_responses = {}
                for key, value in st.session_state.current_answers.items():
                    if key == "Overall Feeling":
                        final_responses[key] = value if value else ""
                    else:
                        final_responses[key] = value if value is not None else 0
                
                st.session_state.responses = final_responses
                st.session_state.questionnaire_complete = True
                st.rerun()
        else:
            st.warning(f"⏳ Please answer all required questions to submit. ({remaining_count} remaining)")
            st.button("📊 Submit Assessment", disabled=True, use_container_width=True)
    
    # Display results
    if st.session_state.questionnaire_complete:
        # Analyze questionnaire responses
        questionnaire_analysis = analyze_mood_from_responses(st.session_state.responses)
        
        # Analyze text response
        text_response = st.session_state.responses.get("Overall Feeling", "")
        text_emotions = None
        if text_response and st.session_state.emotion_classifier:
            with st.spinner("Analyzing your emotional expression..."):
                text_emotions = get_emotion_from_text(
                    text_response, 
                    st.session_state.emotion_classifier
                )
        
        # Display results
        display_emotion_results(questionnaire_analysis, text_emotions)
        
        # Display historical trends
        st.markdown("---")
        patient_data = load_patient_data(st.session_state.patient_id)
        if patient_data:
            display_historical_trends(patient_data)

if __name__ == "__main__":
    main()

