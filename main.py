"""
CYCLESTRONG AI – Unified Training & Recovery Intelligence Backend
=================================================================

A biology-aware training & recovery intelligence system for women athletes.
Combines CYCLESMART AI (cycle-aware training) and RETURNSTRONG (cycle-aware injury recovery).

Startup Command:
    uvicorn main:app --reload

Environment Variables Required:
    GEMINI_API_KEY - Your Google Gemini API key

Author: CYCLESTRONG AI Team
Version: 1.0.0
"""

import os
from datetime import datetime, date
from typing import Dict, Optional, Literal
from enum import Enum

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import google.generativeai as genai

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

# In-memory storage
athletes_db: Dict[str, dict] = {}

# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class CyclePhase(str, Enum):
    MENSTRUAL = "Menstrual"
    FOLLICULAR = "Follicular"
    OVULATION = "Ovulation"
    LUTEAL = "Luteal"

class TrainingLoad(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

class InjurySeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class Mode(str, Enum):
    TRAIN = "TRAIN"
    RECOVER = "RECOVER"

# Phase-based configuration
PHASE_CONFIG = {
    CyclePhase.MENSTRUAL: {
        "base_readiness": 55,
        "injury_risk": "High",
        "recovery_modifier": 1.3,  # Slower recovery
        "training_focus": "Light movement, mobility, and recovery-focused activities",
        "mental_state": "Needs support",
    },
    CyclePhase.FOLLICULAR: {
        "base_readiness": 85,
        "injury_risk": "Low",
        "recovery_modifier": 0.8,  # Faster recovery
        "training_focus": "Progressive overload, skill development, and endurance building",
        "mental_state": "High energy",
    },
    CyclePhase.OVULATION: {
        "base_readiness": 90,
        "injury_risk": "Low",
        "recovery_modifier": 0.9,
        "training_focus": "High-intensity skill and power work",
        "mental_state": "Peak performance",
    },
    CyclePhase.LUTEAL: {
        "base_readiness": 65,
        "injury_risk": "Moderate",
        "recovery_modifier": 1.2,  # Slower recovery
        "training_focus": "Moderate intensity with focus on technique and mental skills",
        "mental_state": "Variable energy",
    },
}

# Training load modifiers
LOAD_MODIFIERS = {
    TrainingLoad.LOW: {"readiness_boost": 10, "injury_risk_reduction": True},
    TrainingLoad.MODERATE: {"readiness_boost": 0, "injury_risk_reduction": False},
    TrainingLoad.HIGH: {"readiness_boost": -10, "injury_risk_reduction": False},
}

# Base recovery days by injury type and severity
BASE_RECOVERY_DAYS = {
    "mild": {"base": 7, "range": (5, 10)},
    "moderate": {"base": 14, "range": (10, 21)},
    "severe": {"base": 28, "range": (21, 42)},
}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class AthleteInput(BaseModel):
    athlete_id: str = Field(..., min_length=1, description="Unique athlete identifier")
    age: int = Field(..., ge=12, le=60, description="Athlete age")
    sport: str = Field(..., min_length=1, description="Sport type")
    cycle_length: int = Field(28, ge=21, le=35, description="Menstrual cycle length in days")
    last_period_date: str = Field(..., description="Last period start date (YYYY-MM-DD)")
    today_date: str = Field(..., description="Current date (YYYY-MM-DD)")

    @field_validator("last_period_date", "today_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

class ModeSelect(BaseModel):
    athlete_id: str = Field(..., min_length=1)
    mode: Mode

class TrainRequest(BaseModel):
    athlete_id: str = Field(..., min_length=1)
    training_load: TrainingLoad

class RecoverRequest(BaseModel):
    athlete_id: str = Field(..., min_length=1)
    injury_type: str = Field(..., min_length=1, description="Type of injury (e.g., hamstring, ankle)")
    injury_severity: InjurySeverity

class HealthResponse(BaseModel):
    status: str

class AthleteResponse(BaseModel):
    message: str
    athlete_id: str
    cycle_day: int
    cycle_phase: str

class ModeResponse(BaseModel):
    message: str
    athlete_id: str
    mode: str

class TrainResponse(BaseModel):
    mode: str = "TRAIN"
    cycle_phase: str
    readiness_score: int
    injury_risk: str
    training_recommendation: str
    coach_message: str

class RecoverResponse(BaseModel):
    mode: str = "RECOVER"
    cycle_phase: str
    estimated_recovery_days: int
    mental_readiness: str
    return_to_play_score: int
    coach_message: str

# =============================================================================
# BIOLOGY LOGIC
# =============================================================================

def calculate_cycle_day(last_period_date: str, today_date: str, cycle_length: int) -> int:
    """
    Calculate the current day in the menstrual cycle.
    Returns a value between 1 and cycle_length.
    """
    last_period = datetime.strptime(last_period_date, "%Y-%m-%d").date()
    today = datetime.strptime(today_date, "%Y-%m-%d").date()
    
    days_since = (today - last_period).days
    
    if days_since < 0:
        raise ValueError("Last period date cannot be in the future")
    
    # Calculate cycle day (1-indexed, wraps around based on cycle length)
    cycle_day = (days_since % cycle_length) + 1
    
    return cycle_day

def get_cycle_phase(cycle_day: int, cycle_length: int = 28) -> CyclePhase:
    """
    Determine the menstrual cycle phase based on cycle day.
    
    Phase Mapping (for 28-day cycle):
    - Day 1-5: Menstrual
    - Day 6-13: Follicular
    - Day 14-16: Ovulation
    - Day 17-28: Luteal
    
    Adjusts proportionally for different cycle lengths.
    """
    # Calculate proportional day for varying cycle lengths
    ratio = cycle_length / 28
    
    menstrual_end = int(5 * ratio)
    follicular_end = int(13 * ratio)
    ovulation_end = int(16 * ratio)
    
    if cycle_day <= menstrual_end:
        return CyclePhase.MENSTRUAL
    elif cycle_day <= follicular_end:
        return CyclePhase.FOLLICULAR
    elif cycle_day <= ovulation_end:
        return CyclePhase.OVULATION
    else:
        return CyclePhase.LUTEAL

def calculate_readiness_score(phase: CyclePhase, training_load: TrainingLoad) -> int:
    """
    Calculate training readiness score (0-100) based on cycle phase and training load.
    """
    base_readiness = PHASE_CONFIG[phase]["base_readiness"]
    load_modifier = LOAD_MODIFIERS[training_load]["readiness_boost"]
    
    # Calculate final score with bounds
    score = base_readiness + load_modifier
    
    # Add small variation based on phase characteristics
    if phase == CyclePhase.OVULATION and training_load == TrainingLoad.HIGH:
        score += 5  # Bonus for high intensity during ovulation
    elif phase == CyclePhase.MENSTRUAL and training_load == TrainingLoad.HIGH:
        score -= 10  # Penalty for high intensity during menstrual
    
    return max(0, min(100, score))

def calculate_injury_risk(phase: CyclePhase, training_load: TrainingLoad) -> str:
    """
    Determine injury risk level based on cycle phase and training load.
    """
    base_risk = PHASE_CONFIG[phase]["injury_risk"]
    
    # Adjust risk based on training load
    if training_load == TrainingLoad.LOW and base_risk == "Moderate":
        return "Low"
    elif training_load == TrainingLoad.HIGH:
        if base_risk == "Low":
            return "Moderate"
        elif base_risk == "Moderate":
            return "High"
    
    return base_risk

def calculate_recovery_days(
    injury_severity: InjurySeverity, 
    phase: CyclePhase,
    injury_type: str
) -> int:
    """
    Calculate estimated recovery days based on injury severity and cycle phase.
    """
    base_days = BASE_RECOVERY_DAYS[injury_severity.value]["base"]
    phase_modifier = PHASE_CONFIG[phase]["recovery_modifier"]
    
    # Calculate adjusted recovery days
    adjusted_days = int(base_days * phase_modifier)
    
    # Add slight variation based on common injury types
    injury_type_lower = injury_type.lower()
    if "muscle" in injury_type_lower or "hamstring" in injury_type_lower or "quad" in injury_type_lower:
        adjusted_days += 2  # Muscle injuries may take slightly longer
    elif "ankle" in injury_type_lower or "wrist" in injury_type_lower:
        adjusted_days -= 1  # Joint injuries with proper care
    
    return max(3, adjusted_days)

def calculate_return_to_play_score(
    injury_severity: InjurySeverity,
    phase: CyclePhase
) -> int:
    """
    Calculate return-to-play confidence score (0-100).
    Higher score = more confident in return timeline.
    """
    # Base scores by severity
    severity_scores = {
        InjurySeverity.MILD: 80,
        InjurySeverity.MODERATE: 60,
        InjurySeverity.SEVERE: 40,
    }
    
    base_score = severity_scores[injury_severity]
    
    # Adjust based on phase
    if phase == CyclePhase.FOLLICULAR:
        base_score += 10  # Faster healing phase
    elif phase in [CyclePhase.LUTEAL, CyclePhase.MENSTRUAL]:
        base_score -= 8  # Slower healing phases
    
    return max(0, min(100, base_score))

# =============================================================================
# AI LOGIC (GEMINI INTEGRATION)
# =============================================================================

async def generate_training_insight(
    phase: CyclePhase,
    readiness_score: int,
    injury_risk: str,
    training_load: TrainingLoad,
    sport: str
) -> tuple[str, str]:
    """
    Generate athlete-facing insight and coach-safe recommendation using Gemini AI.
    Returns: (training_recommendation, coach_message)
    """
    if not gemini_model:
        # Fallback when Gemini is not configured
        return _generate_fallback_training_insight(phase, readiness_score, injury_risk, training_load, sport)
    
    prompt = f"""You are a sports science AI assistant for women athletes. Generate supportive, non-clinical training recommendations.

Context (DO NOT mention these biological details in the output):
- Current training readiness: {readiness_score}/100
- Current injury risk: {injury_risk}
- Requested training intensity: {training_load.value}
- Sport: {sport}

Generate TWO responses:
1. ATHLETE_INSIGHT: A brief, encouraging message (2-3 sentences) about their training capacity today. Be supportive and empowering. Do not mention menstrual cycle or biological details.

2. COACH_MESSAGE: A professional, concise recommendation (2-3 sentences) for the coaching staff about training approach. Focus on performance optimization and injury prevention. Keep it athlete-privacy safe.

Format your response exactly as:
ATHLETE_INSIGHT: [your insight here]
COACH_MESSAGE: [your message here]"""

    try:
        response = await gemini_model.generate_content_async(prompt)
        text = response.text.strip()
        
        # Parse response
        athlete_insight = ""
        coach_message = ""
        
        for line in text.split("\n"):
            if line.startswith("ATHLETE_INSIGHT:"):
                athlete_insight = line.replace("ATHLETE_INSIGHT:", "").strip()
            elif line.startswith("COACH_MESSAGE:"):
                coach_message = line.replace("COACH_MESSAGE:", "").strip()
        
        if athlete_insight and coach_message:
            return athlete_insight, coach_message
        else:
            return _generate_fallback_training_insight(phase, readiness_score, injury_risk, training_load, sport)
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        return _generate_fallback_training_insight(phase, readiness_score, injury_risk, training_load, sport)

def _generate_fallback_training_insight(
    phase: CyclePhase,
    readiness_score: int,
    injury_risk: str,
    training_load: TrainingLoad,
    sport: str
) -> tuple[str, str]:
    """Fallback training insights when Gemini is unavailable."""
    phase_config = PHASE_CONFIG[phase]
    
    if readiness_score >= 80:
        athlete_insight = f"You're in an excellent training window! Your body is primed for {phase_config['training_focus'].lower()}. Push yourself today and make the most of this high-energy period."
        coach_message = f"Athlete is in a high-readiness phase (score: {readiness_score}). Recommend increasing intensity with proper recovery focus. {injury_risk} injury risk - standard precautions advised."
    elif readiness_score >= 60:
        athlete_insight = f"You have good capacity for moderate training today. Focus on {phase_config['training_focus'].lower()}. Listen to your body and adjust as needed."
        coach_message = f"Athlete shows moderate readiness (score: {readiness_score}). Maintain current intensity with attention to form. {injury_risk} injury risk - monitor closely during high-demand drills."
    else:
        athlete_insight = f"Your body is signaling for lighter training today. Focus on {phase_config['training_focus'].lower()}. Recovery is just as important as training."
        coach_message = f"Athlete readiness is lower (score: {readiness_score}). Recommend reducing intensity and focusing on technique/recovery. {injury_risk} injury risk - prioritize warm-up and cool-down."
    
    return athlete_insight, coach_message

async def generate_recovery_insight(
    phase: CyclePhase,
    injury_type: str,
    injury_severity: InjurySeverity,
    recovery_days: int,
    return_to_play_score: int,
    sport: str
) -> str:
    """
    Generate coach-safe recovery messaging using Gemini AI.
    """
    if not gemini_model:
        return _generate_fallback_recovery_insight(phase, injury_type, injury_severity, recovery_days, return_to_play_score)
    
    prompt = f"""You are a sports rehabilitation AI assistant. Generate supportive, professional recovery guidance.

Context (DO NOT mention biological/cycle details):
- Injury type: {injury_type}
- Severity: {injury_severity.value}
- Estimated recovery: {recovery_days} days
- Return-to-play confidence: {return_to_play_score}/100
- Sport: {sport}

Generate a COACH_MESSAGE: A professional, supportive recommendation (2-3 sentences) for the coaching staff about the athlete's recovery approach. Focus on:
- Timeline expectations
- Key recovery priorities
- Mental wellness support needs
- Return-to-play considerations

Keep the message athlete-privacy safe. Do not mention any biological or hormonal details.

Format: COACH_MESSAGE: [your message here]"""

    try:
        response = await gemini_model.generate_content_async(prompt)
        text = response.text.strip()
        
        for line in text.split("\n"):
            if line.startswith("COACH_MESSAGE:"):
                return line.replace("COACH_MESSAGE:", "").strip()
        
        return _generate_fallback_recovery_insight(phase, injury_type, injury_severity, recovery_days, return_to_play_score)
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        return _generate_fallback_recovery_insight(phase, injury_type, injury_severity, recovery_days, return_to_play_score)

def _generate_fallback_recovery_insight(
    phase: CyclePhase,
    injury_type: str,
    injury_severity: InjurySeverity,
    recovery_days: int,
    return_to_play_score: int
) -> str:
    """Fallback recovery insight when Gemini is unavailable."""
    mental_state = PHASE_CONFIG[phase]["mental_state"]
    
    if injury_severity == InjurySeverity.MILD:
        base_message = f"Athlete recovering from mild {injury_type} injury. Expected timeline: {recovery_days} days."
    elif injury_severity == InjurySeverity.MODERATE:
        base_message = f"Athlete in moderate recovery phase for {injury_type} injury. Estimated {recovery_days} days to full return."
    else:
        base_message = f"Athlete undergoing significant recovery for {injury_type} injury. Extended timeline of {recovery_days} days expected."
    
    if mental_state == "Needs support":
        base_message += " Prioritize rest, flexibility, and mental reassurance during this period."
    elif mental_state == "Variable energy":
        base_message += " Energy levels may fluctuate - adjust rehab intensity accordingly."
    else:
        base_message += " Current phase supports active recovery - maintain progressive rehab protocol."
    
    return base_message

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="CYCLESTRONG AI",
    description="Unified Training & Recovery Intelligence for Women Athletes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the backend is running.
    """
    return HealthResponse(status="CYCLESTRONG AI backend running")


@app.post("/athlete/input", response_model=AthleteResponse, tags=["Athlete"])
async def athlete_input(data: AthleteInput):
    """
    Register or update an athlete profile.
    Computes cycle day and phase based on provided dates.
    """
    try:
        # Validate date ordering
        last_period = datetime.strptime(data.last_period_date, "%Y-%m-%d").date()
        today = datetime.strptime(data.today_date, "%Y-%m-%d").date()
        
        if last_period > today:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Last period date cannot be after today's date"
            )
        
        # Calculate cycle information
        cycle_day = calculate_cycle_day(
            data.last_period_date, 
            data.today_date, 
            data.cycle_length
        )
        cycle_phase = get_cycle_phase(cycle_day, data.cycle_length)
        
        # Store athlete data
        athletes_db[data.athlete_id] = {
            "athlete_id": data.athlete_id,
            "age": data.age,
            "sport": data.sport,
            "cycle_length": data.cycle_length,
            "last_period_date": data.last_period_date,
            "today_date": data.today_date,
            "cycle_day": cycle_day,
            "cycle_phase": cycle_phase,
            "mode": None,  # Mode not yet selected
        }
        
        return AthleteResponse(
            message="Athlete profile created successfully",
            athlete_id=data.athlete_id,
            cycle_day=cycle_day,
            cycle_phase=cycle_phase.value
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@app.post("/mode/select", response_model=ModeResponse, tags=["Mode"])
async def mode_select(data: ModeSelect):
    """
    Select the operating mode for an athlete: TRAIN or RECOVER.
    """
    if data.athlete_id not in athletes_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Athlete '{data.athlete_id}' not found. Please register first via /athlete/input"
        )
    
    # Lock athlete into selected mode
    athletes_db[data.athlete_id]["mode"] = data.mode
    
    return ModeResponse(
        message=f"Mode set to {data.mode.value}",
        athlete_id=data.athlete_id,
        mode=data.mode.value
    )


@app.post("/train/recommend", response_model=TrainResponse, tags=["Training"])
async def train_recommend(data: TrainRequest):
    """
    Generate training recommendations based on cycle phase and training load.
    Requires athlete to be in TRAIN mode.
    """
    # Validate athlete exists
    if data.athlete_id not in athletes_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Athlete '{data.athlete_id}' not found. Please register first via /athlete/input"
        )
    
    athlete = athletes_db[data.athlete_id]
    
    # Validate mode
    if athlete["mode"] is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode not selected. Please select a mode via /mode/select first"
        )
    
    if athlete["mode"] != Mode.TRAIN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Athlete is in {athlete['mode'].value} mode. This endpoint requires TRAIN mode. Use /recover/plan for recovery intelligence."
        )
    
    # Calculate training metrics
    phase = athlete["cycle_phase"]
    readiness_score = calculate_readiness_score(phase, data.training_load)
    injury_risk = calculate_injury_risk(phase, data.training_load)
    
    # Generate AI insights
    training_recommendation, coach_message = await generate_training_insight(
        phase=phase,
        readiness_score=readiness_score,
        injury_risk=injury_risk,
        training_load=data.training_load,
        sport=athlete["sport"]
    )
    
    return TrainResponse(
        mode="TRAIN",
        cycle_phase=phase.value,
        readiness_score=readiness_score,
        injury_risk=injury_risk,
        training_recommendation=training_recommendation,
        coach_message=coach_message
    )


@app.post("/recover/plan", response_model=RecoverResponse, tags=["Recovery"])
async def recover_plan(data: RecoverRequest):
    """
    Generate recovery plan based on injury details and cycle phase.
    Requires athlete to be in RECOVER mode.
    """
    # Validate athlete exists
    if data.athlete_id not in athletes_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Athlete '{data.athlete_id}' not found. Please register first via /athlete/input"
        )
    
    athlete = athletes_db[data.athlete_id]
    
    # Validate mode
    if athlete["mode"] is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode not selected. Please select a mode via /mode/select first"
        )
    
    if athlete["mode"] != Mode.RECOVER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Athlete is in {athlete['mode'].value} mode. This endpoint requires RECOVER mode. Use /train/recommend for training intelligence."
        )
    
    # Calculate recovery metrics
    phase = athlete["cycle_phase"]
    recovery_days = calculate_recovery_days(
        injury_severity=data.injury_severity,
        phase=phase,
        injury_type=data.injury_type
    )
    mental_readiness = PHASE_CONFIG[phase]["mental_state"]
    return_to_play_score = calculate_return_to_play_score(
        injury_severity=data.injury_severity,
        phase=phase
    )
    
    # Generate AI insights
    coach_message = await generate_recovery_insight(
        phase=phase,
        injury_type=data.injury_type,
        injury_severity=data.injury_severity,
        recovery_days=recovery_days,
        return_to_play_score=return_to_play_score,
        sport=athlete["sport"]
    )
    
    return RecoverResponse(
        mode="RECOVER",
        cycle_phase=phase.value,
        estimated_recovery_days=recovery_days,
        mental_readiness=mental_readiness,
        return_to_play_score=return_to_play_score,
        coach_message=coach_message
    )


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
