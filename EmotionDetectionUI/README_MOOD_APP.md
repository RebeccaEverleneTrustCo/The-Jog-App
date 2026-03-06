# Patient Mood & Emotion Assessment App

A comprehensive Streamlit application for assessing patient mood and emotions through an interactive questionnaire combined with AI-powered text emotion detection. **Now with historical tracking and trend analysis!**

## Features

### Core Assessment Features
- **Interactive Questionnaire**: 6 multiple-choice questions + 1 optional text response covering:
  - Sleep patterns
  - Energy levels
  - Social interaction preferences
  - Mood stability
  - Stress levels
  - Interest in activities
  
- **Text Emotion Analysis**: Uses HuggingFace's `michellejieli/emotion_text_classifier` model to analyze free-form text responses

- **Comprehensive Results**:
  - Quick summary metrics at a glance
  - Visual mood distribution (pie chart)
  - Detected emotions from text (bar chart)
  - Overall assessment and interpretation
  - Personalized recommendations with positive reinforcement
  - Response review section
  - Exportable report

### 🆕 New: Historical Tracking & Trends
- **Patient Identification**: Track assessments by patient name or ID
- **Multiple Assessments**: Complete the questionnaire multiple times to track mood changes
- **Historical Trends**: Visual line graphs showing mood patterns over time
- **Assessment History**: View all past assessments with timestamps and results
- **Longitudinal Analysis**: Compare current mood to previous assessments
- **File-Based Storage**: Data saved securely in JSON format in `patient_data/` directory

### User Experience Features
- **Informed Consent**: Clear disclaimer screen before starting
- **Dynamic Progress Tracking**: Real-time progress bar showing completion status
- **Collapsible Results**: Organized sections for easy navigation
- **Positive Reinforcement**: Encouraging messages for healthy patterns
- **Supportive Messaging**: Compassionate guidance for concerning patterns
- **Crisis Resources**: Immediate access to mental health support contacts
- **Clean, Modern UI**: Custom styling with mobile-responsive design

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run mood_questionnaire_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Workflow

1. **Patient Identification**
   - Enter patient name or ID (e.g., "John Doe" or "Patient-12345")
   - Or select from existing patients
   - First-time users will be welcomed; returning users will see their assessment count

2. **Informed Consent**
   - Read the disclaimers and assessment purpose
   - Review crisis resources if needed
   - Check the consent checkbox
   - Click "I Agree - Begin Assessment"

3. **Complete Questionnaire**
   - Answer 6 required multiple-choice questions
   - Optionally provide text response for detailed emotion analysis
   - Watch progress bar update in real-time
   - Submit when all required questions are answered

4. **View Results**
   - See current assessment results with visualizations
   - Review personalized recommendations
   - View historical trends (if multiple assessments completed)
   - Export report if needed

5. **Track Over Time**
   - Click "Start New Assessment" to take another assessment
   - Use "View Historical Trends" to see mood changes over time
   - Compare current results with past assessments

## Data Storage

### File Structure
```
RebeccaLandAI/
├── mood_questionnaire_app.py
├── requirements.txt
├── patient_data/          # Created automatically
│   ├── John_Doe.json
│   ├── Patient-12345.json
│   └── ...
```

### Patient Data Format
Each patient has a JSON file containing:
```json
{
  "patient_id": "John_Doe",
  "assessments": [
    {
      "timestamp": "2026-02-01T14:30:00",
      "responses": { ... },
      "mood_analysis": {
        "positive": 0.65,
        "neutral": 0.25,
        "negative": 0.10
      },
      "text_emotions": [ ... ]
    }
  ]
}
```

### Privacy & Security
- Data is stored locally in the `patient_data/` directory
- No data is sent to external servers (except for model inference during session)
- Files are in plain JSON format for transparency
- Add `patient_data/` to `.gitignore` to prevent committing sensitive data
- For production use, consider encryption and secure storage solutions

## Historical Trends Features

### Mood Trends Over Time
- **Line Graph**: Shows positive, neutral, and negative mood percentages across all assessments
- **Interactive**: Hover to see exact values for each date
- **Color Coded**: Green (positive), Gray (neutral), Red (negative)

### Assessment History
- **Total Count**: Number of completed assessments
- **Date Range**: First and latest assessment dates
- **Detailed View**: Expandable list showing all past assessments with:
  - Date and time
  - Dominant mood
  - Mood percentages

### Trend Analysis Benefits
- **Track Progress**: See if interventions or life changes are improving mood
- **Identify Patterns**: Spot recurring mood fluctuations
- **Clinical Value**: Share trends with healthcare providers
- **Self-Awareness**: Understand long-term emotional patterns

1. **Quick Summary** (always visible)
   - Dominant mood at a glance
   - Primary detected emotion
   - Number of questions answered

2. **Questionnaire Analysis** (expandable)
   - Mood distribution pie chart
   - Detailed percentage breakdown

3. **Text Emotion Analysis** (expandable)
   - Emotion confidence bar chart
   - Top 5 emotion scores

4. **Overall Assessment** (expandable)
   - Interpretation of results
   - Key insights for each area (sleep, energy, stress, social)

5. **Personalized Recommendations** (expandable)
   - Action items based on responses

6. **Your Responses Summary** (expandable)
   - Review all your answers

This organization allows users to focus on one insight at a time without feeling overwhelmed.

## How It Works

### 1. Questionnaire Analysis
- Each multiple-choice response is scored
- Responses are categorized into positive, neutral, and negative mood indicators
- A weighted score is calculated based on all responses

### 2. Text Emotion Detection
- Uses the `michellejieli/emotion_text_classifier` model
- Analyzes the patient's free-form text response
- Detects multiple emotions with confidence scores
- Identifies the primary emotion

### 3. Results Integration
- Combines questionnaire scores with text emotion analysis
- Provides holistic mood assessment
- Generates personalized recommendations

## Model Information

**Emotion Classifier**: `michellejieli/emotion_text_classifier`
- Lightweight and efficient
- Detects multiple emotions: sadness, joy, love, anger, fear, surprise
- Provides confidence scores for each emotion

## Important Note

This tool is for assessment and informational purposes only. It is NOT a diagnostic tool and should not replace professional mental health evaluation. Users experiencing significant distress should consult with a qualified mental health professional.

## Customization

You can easily customize:
- Questions in the `QUESTIONNAIRE` dictionary
- Scoring logic in `analyze_mood_from_responses()`
- Recommendations in `generate_recommendations()`
- Visual styling in the CSS section

## Export Feature

Users can export their assessment results as a text report including:
- Timestamp
- Questionnaire analysis scores
- Text emotion detection results
- All responses

## Technologies Used

- **Streamlit**: Web interface framework
- **Transformers (HuggingFace)**: Emotion detection model
- **Plotly**: Interactive visualizations
- **PyTorch**: Model inference backend
- **Pandas**: Data manipulation

