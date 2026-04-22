import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re

# We will load models lazily to avoid heavy start times if not needed immediately
nlp = None
vader = None
mental_bert = None

def load_models():
    global nlp, vader, mental_bert
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            print("Downloading spacy model en_core_web_sm...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            
    if vader is None:
        vader = SentimentIntensityAnalyzer()
        
    if mental_bert is None:
        # MentalBERT from huggingface, using typical base uncased if mental-bert directly fails
        try:
            mental_bert = pipeline("text-classification", model="mental/mental-bert-base-uncased")
        except Exception as e:
            print(f"Failed to load mental-bert, falling back to default sentiment. Error: {e}")
            mental_bert = pipeline("sentiment-analysis")

def extract_linguistic_features(text: str) -> dict:
    load_models()
    doc = nlp(text)
    
    anxiety_words = {"stress", "anxious", "overwhelmed", "panic", "worry", "tired", "exhausted", "fear", "hard", "struggle", "bad", "sad"}
    negation_words = {"not", "never", "no", "none", "cannot", "can't", "don't", "won't"}
    
    anxiety_count = sum(1 for token in doc if token.lemma_.lower() in anxiety_words)
    negation_count = sum(1 for token in doc if token.lemma_.lower() in negation_words)
    sentence_complexity = sum(1 for token in doc if token.pos_ in ("SCONJ", "CCONJ")) / max(1, len(list(doc.sents)))
    
    return {
        "anxiety_word_count": anxiety_count,
        "negation_count": negation_count,
        "sentence_complexity": round(sentence_complexity, 2),
        "total_words": len(doc)
    }

def analyze_stress_from_text(text: str) -> dict:
    """Combines MentalBERT, SpaCy, and VADER to classify text stress"""
    load_models()
    
    # NLP features
    ling_features = extract_linguistic_features(text)
    
    # VADER fallback/complement
    vader_scores = vader.polarity_scores(text)
    compound = vader_scores["compound"]
    
    # MentalBERT classification
    bert_results = mental_bert(text)
    
    # Map poor sentiment / negative mental-bert back to high stress
    # Vader: -1 is max negative, 1 is max positive
    stress_score = 0.5 - (compound * 0.5)
    
    # Adjust based on anxiety words (+0.1 per word, max +0.4)
    stress_score += min(0.4, ling_features["anxiety_word_count"] * 0.1)
    
    # Clamp to valid range [0, 1]
    stress_score = max(0.0, min(1.0, stress_score))
    
    if stress_score < 0.35:
        tier = "Low"
    elif stress_score < 0.7:
        tier = "Moderate"
    else:
        tier = "High"
        
    return {
        "stress_score": stress_score,
        "stress_tier": tier,
        "linguistic_features": ling_features,
        "bert_raw": bert_results
    }
