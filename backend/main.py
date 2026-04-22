from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import math
import json
import datetime

from backend.database import engine, get_db, Base
from backend import models, encryption
from backend.ml.text_pipeline import analyze_stress_from_text
from backend.ml.physio_pipeline import analyze_physiological_stress
from backend.ml.rag_pipeline import generate_rag_response

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="MAITRI API", description="Backend for MAITRI mental health integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str
    physio_features: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    stress_tier: str
    stress_score: float
    shap_explanations: Optional[list] = None

class JournalCreate(BaseModel):
    content: str
    mood_tag: str

class JournalResponse(BaseModel):
    id: int
    content: str
    mood_tag: str
    timestamp: datetime.datetime

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "MAITRI Core"}

@app.post("/chat", response_model=ChatResponse)
def process_chat(req: ChatRequest, db: Session = Depends(get_db)):
    # 1. Analyze text stress
    text_analysis = analyze_stress_from_text(req.message)
    stress_tier = text_analysis["stress_tier"]
    stress_score = text_analysis["stress_score"]
    
    # 2. Analyze physiological stress if features provided
    shap_explanations = None
    if req.physio_features:
        physio_analysis = analyze_physiological_stress(req.physio_features)
        if physio_analysis["is_stressed"] and stress_tier != "High":
            stress_tier = "Moderate" if stress_tier == "Low" else "High"
            stress_score = max(stress_score, physio_analysis["stress_probability"])
        shap_explanations = physio_analysis["shap_explanations"]
        
    # 3. Pull decrypted Journals to relate Chat to Journal
    recent_journals = db.query(models.JournalEntry).order_by(models.JournalEntry.timestamp.desc()).limit(3).all()
    journal_context = ""
    if recent_journals:
        journal_context = "### User's Recent Journal Entries (for psychological context):\n"
        for j in recent_journals:
            dec = encryption.decrypt_text(j.encrypted_content)
            journal_context += f"- [{j.timestamp.strftime('%Y-%m-%d')}] Tag: {j.mood_tag}. Note: {dec}\n"
            
    recent_chat = db.query(models.Message).order_by(models.Message.timestamp.desc()).limit(8).all()
    chat_history = ""
    if recent_chat:
        chat_history = "### Recent Conversation History:\n"
        for m in reversed(recent_chat):
            chat_history += f"[{'MAITRI' if m.role == 'agent' else 'Astronaut'}]: {m.content}\n"
    
    # 4. Generate RAG Response
    ai_response = generate_rag_response(req.message, stress_tier, journal_context, chat_history)
    
    # 5. Save to Database
    user_msg = models.Message(
        role="user",
        content=req.message,
        stress_score=stress_score,
        stress_tier=stress_tier,
        shap_data=json.dumps(shap_explanations) if shap_explanations else None
    )
    db.add(user_msg)
    
    agent_msg = models.Message(
        role="agent",
        content=ai_response
    )
    db.add(agent_msg)
    db.commit()
    
    return ChatResponse(
        response=ai_response,
        stress_tier=stress_tier,
        stress_score=stress_score,
        shap_explanations=shap_explanations
    )

@app.post("/journal", response_model=JournalResponse)
def create_journal(entry: JournalCreate, db: Session = Depends(get_db)):
    enc_content = encryption.encrypt_text(entry.content)
    db_entry = models.JournalEntry(
        encrypted_content=enc_content,
        mood_tag=entry.mood_tag
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    return JournalResponse(
        id=db_entry.id,
        content=entry.content,
        mood_tag=db_entry.mood_tag,
        timestamp=db_entry.timestamp
    )

@app.get("/journal", response_model=List[JournalResponse])
def get_journals(db: Session = Depends(get_db)):
    entries = db.query(models.JournalEntry).order_by(models.JournalEntry.timestamp.desc()).limit(50).all()
    results = []
    for e in entries:
        dec_content = encryption.decrypt_text(e.encrypted_content)
        results.append(JournalResponse(
            id=e.id,
            content=dec_content,
            mood_tag=e.mood_tag,
            timestamp=e.timestamp
        ))
    return results

@app.get("/journal/summary")
def get_journal_summary(db: Session = Depends(get_db)):
    """Replay your week: Prompts LLM to summarize the user's encrypted journals over the past 7 days."""
    try:
        recent_journals = db.query(models.JournalEntry).order_by(models.JournalEntry.timestamp.desc()).limit(10).all()
        if not recent_journals:
            return {"summary": "No journal entries found to replay yet. Start logging your thoughts to see your weekly psychological summary."}
            
        combined_text = ""
        for j in recent_journals:
            dec = encryption.decrypt_text(j.encrypted_content)
            combined_text += f"[{j.timestamp.strftime('%a %I:%M%p')}] Mood: {j.mood_tag}. Note: {dec}\n"
        
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3:latest")
        system_prompt = (
            "You are MAITRI, an empathetic AI space psychologist. Analyze the following psychological logs from the astronaut's past week. "
            "Write a beautiful, highly encouraging 'Week in Review' paragraph highlighting their emotional journey, acknowledging any struggles they faced, and showing pride in their resilience. "
            "Keep it under 4 sentences. Address them directly as 'you'. Logs:\n" + combined_text
        )
        summary = llm.invoke(system_prompt)
        return {"summary": summary}
    except Exception as e:
        print("Summary Error:", e)
        return {"summary": "Unable to decrypt and summarize neural history at this time."}

@app.get("/trends")
def get_trends(db: Session = Depends(get_db)):
    messages = db.query(models.Message).filter(models.Message.role == "user").order_by(models.Message.timestamp.desc()).limit(100).all()
    
    trend_data = {}
    for msg in messages:
        if not msg.stress_score:
            continue
        day_str = msg.timestamp.strftime("%m-%d")
        if day_str not in trend_data:
            trend_data[day_str] = []
        trend_data[day_str].append(msg.stress_score)
        
    formatted_trends = []
    for day in sorted(trend_data.keys()):
        avg_score = sum(trend_data[day]) / len(trend_data[day])
        formatted_trends.append({"date": day, "avg_stress": round(avg_score, 3)})
        
    return {"trends": formatted_trends}

@app.get("/interventions")
def get_interventions(stress_tier: str = "Low"):
    if stress_tier == "High":
        return {"recommendations": [
            "Initiate 5-minute Progressive Muscle Relaxation (PMR).",
            "Mandate 9-hour dark period for sleep hygiene.",
            "Contact Medical Officer if symptoms persist."
        ]}
    elif stress_tier == "Moderate":
        return {"recommendations": [
            "Review cognitive reframing techniques.",
            "Draft asynchronous audio message to family.",
            "Take a 10-minute visual break."
        ]}
    else:
        return {"recommendations": [
            "Maintain current workload and schedule.",
            "Continue regular journal logging."
        ]}
