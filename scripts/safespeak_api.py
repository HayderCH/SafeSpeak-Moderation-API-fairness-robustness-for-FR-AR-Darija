#!/usr/bin/env python3
"""
SafeSpeak API v1.0
Production-ready REST API for multilingual toxicity detection

Features:
- FastAPI-based REST endpoints
- Integrated guardrails and monitoring
- Batch processing support
- Health checks and metrics
- Production logging and error handling
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.production_guardrails import ProductionGuardrails

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# API Models
class PredictionRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(
        ..., min_length=1, max_length=1000, description="Text to classify for toxicity"
    )
    user_id: Optional[str] = Field(
        None, description="User identifier for rate limiting"
    )
    request_id: Optional[str] = Field(None, description="Custom request identifier")

    @validator("text")
    def validate_text_length(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of texts to classify"
    )
    user_id: Optional[str] = Field(
        None, description="User identifier for rate limiting"
    )
    request_id: Optional[str] = Field(None, description="Custom request identifier")


class PredictionResponse(BaseModel):
    """Prediction response."""

    request_id: str
    timestamp: str
    success: bool
    prediction: Optional[int] = None
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    is_fallback: bool = False
    processing_time: float
    language: Optional[str] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    request_id: str
    timestamp: str
    results: List[PredictionResponse]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    uptime: float
    components: Dict[str, Any]


# Global variables
app = FastAPI(
    title="SafeSpeak API",
    description="Multilingual Toxicity Detection & Moderation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
guardrails: Optional[ProductionGuardrails] = None
start_time = time.time()


def load_model():
    """Load the production model and initialize guardrails."""
    global guardrails

    try:
        # Load model and tokenizer
        model_path = "results/bert_max_french_augmentation/fold_0/checkpoint-9836"
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, return_dict=False
        )
        model.eval()

        # Initialize guardrails
        guardrails = ProductionGuardrails(
            model=model,
            tokenizer=tokenizer,
            enable_rate_limiting=True,
            enable_privacy_logging=True,
            enable_circuit_breaker=True,
        )

        logger.info("Model and guardrails loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Initialize guardrails without model (fallback mode)
        guardrails = ProductionGuardrails(
            model=None,
            tokenizer=None,
            enable_rate_limiting=True,
            enable_privacy_logging=True,
            enable_circuit_breaker=True,
        )
        logger.warning("Running in fallback mode without model")


# Load model on startup
load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not guardrails:
        raise HTTPException(status_code=503, detail="Service unavailable")

    health_status = guardrails.get_health_status()

    return HealthResponse(
        status=(
            "healthy" if health_status["overall_health"] == "healthy" else "unhealthy"
        ),
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime=time.time() - start_time,
        components=health_status["components"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(request: PredictionRequest, req: Request):
    """Predict toxicity for a single text."""
    if not guardrails:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Extract client IP for rate limiting
        client_ip = req.client.host if req.client else "unknown"
        user_id = request.user_id or client_ip

        # Process request through guardrails
        result = guardrails.process_request(
            text=request.text, user_id=user_id, request_id=request.request_id
        )

        # Convert to response model
        response = PredictionResponse(
            request_id=result["request_id"],
            timestamp=result["timestamp"],
            success=result["success"],
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            probabilities=result.get("probabilities"),
            is_fallback=result.get("is_fallback", False),
            processing_time=result["processing_time"],
            language=result["validation"]["metadata"].get("language"),
            error=result.get("error"),
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_toxicity_batch(request: BatchPredictionRequest, req: Request):
    """Predict toxicity for multiple texts."""
    if not guardrails:
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Extract client IP for rate limiting
        client_ip = req.client.host if req.client else "unknown"
        user_id = request.user_id or client_ip

        # Process each text
        results = []
        total_toxic = 0
        avg_confidence = 0.0

        for text in request.texts:
            result = guardrails.process_request(
                text=text, user_id=user_id, request_id=request.request_id
            )

            response = PredictionResponse(
                request_id=result["request_id"],
                timestamp=result["timestamp"],
                success=result["success"],
                prediction=result.get("prediction"),
                confidence=result.get("confidence"),
                probabilities=result.get("probabilities"),
                is_fallback=result.get("is_fallback", False),
                processing_time=result["processing_time"],
                language=result["validation"]["metadata"].get("language"),
                error=result.get("error"),
            )

            results.append(response)

            if response.prediction == 1:  # Assuming 1 = toxic
                total_toxic += 1
            if response.confidence:
                avg_confidence += response.confidence

        if results:
            avg_confidence /= len(results)

        summary = {
            "total_texts": len(request.texts),
            "toxic_count": total_toxic,
            "non_toxic_count": len(request.texts) - total_toxic,
            "toxicity_rate": total_toxic / len(request.texts) if request.texts else 0,
            "average_confidence": avg_confidence,
            "fallback_count": sum(1 for r in results if r.is_fallback),
        }

        return BatchPredictionResponse(
            request_id=request.request_id or f"batch_{int(time.time() * 1000)}",
            timestamp=datetime.now().isoformat(),
            results=results,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats")
async def get_usage_stats():
    """Get usage statistics."""
    if not guardrails or not hasattr(guardrails, "privacy_logger"):
        raise HTTPException(status_code=503, detail="Statistics unavailable")

    try:
        stats = guardrails.privacy_logger.get_usage_stats(days=7)
        return {
            "period_days": 7,
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Statistics error")


@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (admin endpoint)."""
    try:
        background_tasks.add_task(load_model)
        return {
            "message": "Model reload initiated",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(status_code=500, detail="Reload failed")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
