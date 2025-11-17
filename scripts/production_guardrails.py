#!/usr/bin/env python3
"""
SafeSpeak Production Guardrails
Phase 4 Platinum: Production safety mechanisms and input validation

This module implements:
- Input validation and sanitization
- Rate limiting and abuse prevention
- Fallback mechanisms for model failures
- Privacy-preserving logging
- Safety thresholds and circuit breakers
"""

import os
import json
import logging
import time
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
import threading
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validation for SafeSpeak toxicity detection.

    Features:
    - Text length validation
    - Language detection and filtering
    - Content filtering for harmful patterns
    - Unicode and encoding validation
    """

    def __init__(
        self,
        max_length: int = 512,
        min_length: int = 1,
        blocked_patterns: List[str] = None,
    ):
        """
        Initialize input validator.

        Args:
            max_length: Maximum text length in characters
            min_length: Minimum text length in characters
            blocked_patterns: List of regex patterns to block
        """
        self.max_length = max_length
        self.min_length = min_length
        self.blocked_patterns = blocked_patterns or [
            r"<script[^>]*>.*?</script>",  # Script tags
            r"<[^>]+>",  # HTML tags
            r"javascript:",  # JavaScript URLs
            r"data:",  # Data URLs
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP addresses
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
        ]

        # Compile regex patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.blocked_patterns
        ]

        # Load language detection for logging/analytics only
        self._load_language_detector()

    def _load_language_detector(self):
        """Load simple language detection heuristics."""
        # Arabic character ranges
        self.arabic_chars = set()
        for start, end in [
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ]:
            self.arabic_chars.update(range(start, end + 1))

        # French common words
        self.french_words = {
            "le",
            "la",
            "les",
            "de",
            "du",
            "des",
            "et",
            "à",
            "un",
            "une",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "être",
            "avoir",
            "faire",
            "aller",
            "venir",
            "voir",
            "savoir",
        }

        # Darija indicators (Arabic with Latin script)
        self.darija_indicators = {"3", "7", "9", "2", "5", "8", "ch", "kh", "gh"}

    def validate_text(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate input text for safety and compliance.

        Args:
            text: Input text to validate

        Returns:
            Tuple of (is_valid, reason, metadata)
        """
        metadata = {
            "length": len(text),
            "language": "unknown",
            "has_blocked_content": False,
            "encoding_valid": True,
            "warnings": [],
        }

        # Basic type and null checks
        if not isinstance(text, str):
            return False, "Input must be a string", metadata

        if not text or text.isspace():
            return False, "Input cannot be empty or whitespace only", metadata

        # Length validation
        if len(text) < self.min_length:
            return (
                False,
                f"Text too short (minimum {self.min_length} characters)",
                metadata,
            )

        if len(text) > self.max_length:
            return (
                False,
                f"Text too long (maximum {self.max_length} characters)",
                metadata,
            )

        # Encoding validation
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            metadata["encoding_valid"] = False
            return False, "Invalid UTF-8 encoding", metadata

        # Blocked content check
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                metadata["has_blocked_content"] = True
                return False, "Content contains blocked patterns", metadata

        # Language detection (for logging/analytics only - no blocking)
        detected_lang = self._detect_language(text)
        metadata["language"] = detected_lang

        # Note: We don't block by language since this is a multilingual toxicity detector
        # All languages are allowed, but we log the detected language for analytics

        # Additional safety checks
        if self._contains_suspicious_patterns(text):
            metadata["warnings"].append("Contains suspicious patterns")

        # Check for excessive repetition
        if self._has_excessive_repetition(text):
            metadata["warnings"].append("Excessive character repetition detected")

        return True, "Valid", metadata

    def _detect_language(self, text: str) -> str:
        """Simple language detection heuristic."""
        text_lower = text.lower()

        # Count Arabic characters
        arabic_count = sum(1 for char in text if ord(char) in self.arabic_chars)

        # Count French words
        french_word_count = sum(
            1 for word in text_lower.split() if word in self.french_words
        )

        # Check for Darija indicators
        darija_score = sum(
            1 for indicator in self.darija_indicators if indicator in text_lower
        )

        # Determine language
        if arabic_count > len(text) * 0.3:  # 30% Arabic characters
            if darija_score > 2:
                return "darija"
            else:
                return "ar"
        elif french_word_count > 2:
            return "fr"
        else:
            return "en"  # Default to English

    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns that might indicate attacks."""
        suspicious = [
            " or 1=1",  # SQL injection
            "<script",  # XSS
            "javascript:",  # JS injection
            "data:text/html",  # Data URL
            "\x00",  # Null bytes
            "\r\n\r\n",  # HTTP headers
        ]

        text_lower = text.lower()
        return any(pattern in text_lower for pattern in suspicious)

    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition."""
        # Check for same character repeated more than 10 times
        for char in set(text):
            if text.count(char) > 10:
                return True
        return False

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing or replacing harmful content.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text


class RateLimiter:
    """
    Token bucket rate limiter for API protection.

    Features:
    - Configurable rates per user/IP
    - Burst handling
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        cleanup_interval: int = 300,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_limit: Maximum burst requests
            cleanup_interval: Cleanup interval in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.cleanup_interval = cleanup_interval

        # Token bucket storage: {identifier: deque of timestamps}
        self.requests = defaultdict(lambda: deque(maxlen=burst_limit))

        # Threading lock for thread safety
        self.lock = threading.Lock()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def is_allowed(self, identifier: str) -> Tuple[bool, float]:
        """
        Check if request is allowed for given identifier.

        Args:
            identifier: User ID, IP address, or other identifier

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        with self.lock:
            now = time.time()
            request_times = self.requests[identifier]

            # Remove old requests outside the time window
            window_start = now - 60  # 1 minute window
            while request_times and request_times[0] < window_start:
                request_times.popleft()

            # Check rate limit
            if len(request_times) >= self.requests_per_minute:
                # Calculate retry time
                oldest_request = request_times[0]
                retry_after = 60 - (now - oldest_request)
                return False, max(0, retry_after)

            # Add current request
            request_times.append(now)
            return True, 0.0

    def _cleanup_worker(self):
        """Background worker to clean up old entries."""
        while True:
            time.sleep(self.cleanup_interval)
            self._cleanup_old_entries()

    def _cleanup_old_entries(self):
        """Remove entries that haven't been active recently."""
        with self.lock:
            now = time.time()
            cutoff = now - 3600  # 1 hour ago

            to_remove = []
            for identifier, request_times in self.requests.items():
                if request_times and request_times[-1] < cutoff:
                    to_remove.append(identifier)

            for identifier in to_remove:
                del self.requests[identifier]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old rate limit entries")


class PrivacyLogger:
    """
    Privacy-preserving logging system for SafeSpeak.

    Features:
    - Log aggregation and anonymization
    - PII detection and redaction
    - Configurable retention policies
    - Compliance with privacy regulations
    """

    def __init__(
        self,
        log_dir: str = "logs/privacy",
        retention_days: int = 90,
        enable_pii_detection: bool = True,
    ):
        """
        Initialize privacy logger.

        Args:
            log_dir: Directory to store logs
            retention_days: Days to retain logs
            enable_pii_detection: Whether to detect and redact PII
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.retention_days = retention_days
        self.enable_pii_detection = enable_pii_detection

        # PII detection patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        }

        # Compile patterns
        self.compiled_pii_patterns = {
            name: re.compile(pattern) for name, pattern in self.pii_patterns.items()
        }

    def log_request(
        self,
        request_id: str,
        user_id: Optional[str],
        input_text: str,
        prediction: Any,
        confidence: float,
        processing_time: float,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Log a prediction request with privacy protection.

        Args:
            request_id: Unique request identifier
            user_id: Anonymized user identifier (optional)
            input_text: Input text (will be redacted)
            prediction: Model prediction
            confidence: Prediction confidence
            processing_time: Processing time in seconds
            metadata: Additional metadata
        """
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "user_id": self._anonymize_user_id(user_id) if user_id else None,
            "input_hash": self._hash_text(input_text),
            "input_length": len(input_text),
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time,
            "metadata": self._sanitize_metadata(metadata),
            "pii_detected": (
                self._detect_pii(input_text) if self.enable_pii_detection else []
            ),
        }

        # Write to daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"requests_{date_str}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID using hashing."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def _hash_text(self, text: str) -> str:
        """Create hash of input text for duplicate detection."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _detect_pii(self, text: str) -> List[str]:
        """Detect PII in text."""
        detected = []
        for pii_type, pattern in self.compiled_pii_patterns.items():
            if pattern.search(text):
                detected.append(pii_type)
        return detected

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata by removing sensitive information."""
        sanitized = metadata.copy()

        # Remove IP addresses from metadata
        if "ip_address" in sanitized:
            sanitized["ip_address"] = self._anonymize_user_id(sanitized["ip_address"])

        # Remove user agent details that might contain PII
        if "user_agent" in sanitized:
            # Keep only browser/OS info, remove specific versions that might identify users
            sanitized["user_agent"] = re.sub(
                r"\d+\.\d+", "X.X", sanitized["user_agent"]
            )

        return sanitized

    def cleanup_old_logs(self) -> int:
        """
        Clean up logs older than retention period.

        Returns:
            Number of files removed
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        removed_count = 0

        for log_file in self.log_dir.glob("*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")

                if file_date < cutoff_date:
                    log_file.unlink()
                    removed_count += 1
            except (ValueError, IndexError):
                # Skip files with unexpected naming
                continue

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old log files")

        return removed_count

    def get_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get usage statistics for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with usage statistics
        """
        stats = {
            "total_requests": 0,
            "unique_users": set(),
            "avg_processing_time": 0.0,
            "avg_confidence": 0.0,
            "pii_detection_rate": 0.0,
            "language_distribution": defaultdict(int),
        }

        cutoff_date = datetime.now() - timedelta(days=days)
        processing_times = []
        confidences = []
        pii_detections = 0

        for log_file in self.log_dir.glob("*.jsonl"):
            try:
                date_str = log_file.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")

                if file_date >= cutoff_date:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                stats["total_requests"] += 1

                                if entry.get("user_id"):
                                    stats["unique_users"].add(entry["user_id"])

                                if entry.get("processing_time"):
                                    processing_times.append(entry["processing_time"])

                                if entry.get("confidence"):
                                    confidences.append(entry["confidence"])

                                if (
                                    entry.get("pii_detected")
                                    and len(entry["pii_detected"]) > 0
                                ):
                                    pii_detections += 1

                                if entry.get("metadata", {}).get("language"):
                                    stats["language_distribution"][
                                        entry["metadata"]["language"]
                                    ] += 1

                            except json.JSONDecodeError:
                                continue

            except (ValueError, IndexError):
                continue

        # Calculate averages
        if processing_times:
            stats["avg_processing_time"] = sum(processing_times) / len(processing_times)

        if confidences:
            stats["avg_confidence"] = sum(confidences) / len(confidences)

        if stats["total_requests"] > 0:
            stats["pii_detection_rate"] = pii_detections / stats["total_requests"]

        stats["unique_users"] = len(stats["unique_users"])

        return dict(stats)


class CircuitBreaker:
    """
    Circuit breaker pattern for service protection.

    Features:
    - Automatic failure detection
    - Graceful degradation
    - Recovery mechanisms
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result or fallback value

        Raises:
            CircuitBreakerOpen: When circuit is open
        """
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpen("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout

    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info("Circuit breaker reset to CLOSED state")

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class ProductionGuardrails:
    """
    Complete production guardrails system for SafeSpeak.

    Integrates all safety mechanisms:
    - Input validation
    - Rate limiting
    - Privacy logging
    - Circuit breaker
    - Fallback mechanisms
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        enable_rate_limiting: bool = True,
        enable_privacy_logging: bool = True,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize production guardrails.

        Args:
            model: ML model for predictions
            tokenizer: Tokenizer for text processing
            enable_rate_limiting: Whether to enable rate limiting
            enable_privacy_logging: Whether to enable privacy logging
            enable_circuit_breaker: Whether to enable circuit breaker
        """
        self.model = model
        self.tokenizer = tokenizer

        # Initialize components
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.privacy_logger = PrivacyLogger() if enable_privacy_logging else None
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        # Fallback responses
        self.fallback_response = {
            "prediction": 0,  # Non-toxic
            "confidence": 0.5,
            "is_fallback": True,
            "reason": "Service temporarily unavailable",
        }

    def process_request(
        self, text: str, user_id: Optional[str] = None, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a prediction request with full guardrails.

        Args:
            text: Input text to classify
            user_id: User identifier for rate limiting
            request_id: Request identifier for logging

        Returns:
            Prediction result with metadata
        """
        start_time = time.time()
        request_id = request_id or f"req_{int(time.time() * 1000)}"

        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "is_fallback": False,
            "processing_time": 0.0,
            "validation": {},
            "rate_limit": {},
            "prediction": None,
            "confidence": None,
            "error": None,
        }

        try:
            # 1. Input validation
            is_valid, reason, validation_metadata = self.validator.validate_text(text)
            result["validation"] = {
                "is_valid": is_valid,
                "reason": reason,
                "metadata": validation_metadata,
            }

            if not is_valid:
                result["error"] = f"Validation failed: {reason}"
                return result

            # 2. Rate limiting
            if self.rate_limiter and user_id:
                allowed, retry_after = self.rate_limiter.is_allowed(user_id)
                result["rate_limit"] = {"allowed": allowed, "retry_after": retry_after}

                if not allowed:
                    result["error"] = (
                        f"Rate limit exceeded. Retry after {retry_after:.1f} seconds"
                    )
                    return result

            # 3. Sanitize input
            sanitized_text = self.validator.sanitize_text(text)

            # 4. Make prediction with circuit breaker
            if self.circuit_breaker:
                prediction_result = self.circuit_breaker.call(
                    self._make_prediction, sanitized_text
                )
            else:
                prediction_result = self._make_prediction(sanitized_text)

            # 5. Update result
            result.update(prediction_result)
            result["success"] = True
            result["processing_time"] = time.time() - start_time

        except CircuitBreakerOpen:
            # Circuit breaker is open, use fallback
            result.update(self.fallback_response)
            result["processing_time"] = time.time() - start_time
            logger.warning(f"Circuit breaker open for request {request_id}")

        except Exception as e:
            # Unexpected error, use fallback
            result.update(self.fallback_response)
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
            logger.error(f"Unexpected error in request {request_id}: {e}")

        finally:
            # 6. Privacy logging (log even failed requests)
            if self.privacy_logger:
                self.privacy_logger.log_request(
                    request_id=request_id,
                    user_id=user_id,
                    input_text=text,
                    prediction=result.get("prediction"),
                    confidence=result.get("confidence", 0.0),
                    processing_time=result["processing_time"],
                    metadata={
                        "validation": result["validation"],
                        "rate_limit": result["rate_limit"],
                        "is_fallback": result.get("is_fallback", False),
                        "error": result.get("error"),
                        "language": result["validation"]
                        .get("metadata", {})
                        .get("language", "unknown"),
                    },
                )

        return result

    def _make_prediction(self, text: str) -> Dict[str, Any]:
        """
        Make prediction using the model.

        Args:
            text: Sanitized input text

        Returns:
            Prediction result
        """
        if not self.model or not self.tokenizer:
            raise Exception("Model or tokenizer not available")

        # Tokenize input
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()[0]

        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs.tolist(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all guardrails components.

        Returns:
            Health status dictionary
        """
        status = {
            "overall_health": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Check circuit breaker
        if self.circuit_breaker:
            status["components"]["circuit_breaker"] = {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "healthy": self.circuit_breaker.state != "OPEN",
            }
            if self.circuit_breaker.state == "OPEN":
                status["overall_health"] = "degraded"

        # Check rate limiter
        if self.rate_limiter:
            status["components"]["rate_limiter"] = {
                "active_users": len(self.rate_limiter.requests),
                "healthy": True,
            }

        # Check privacy logger
        if self.privacy_logger:
            recent_stats = self.privacy_logger.get_usage_stats(days=1)
            status["components"]["privacy_logger"] = {
                "recent_requests": recent_stats.get("total_requests", 0),
                "healthy": True,
            }

        # Check model availability
        status["components"]["model"] = {
            "available": self.model is not None,
            "healthy": self.model is not None,
        }
        if not self.model:
            status["overall_health"] = "unhealthy"

        return status


def main():
    """Demo of production guardrails."""
    import argparse

    parser = argparse.ArgumentParser(description="SafeSpeak Production Guardrails Demo")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--user-id", type=str, default="demo_user", help="User ID")
    parser.add_argument(
        "--health-check", action="store_true", help="Show health status"
    )

    args = parser.parse_args()

    # Initialize guardrails (without actual model for demo)
    guardrails = ProductionGuardrails(
        model=None,  # Would load actual model in production
        tokenizer=None,
        enable_rate_limiting=True,
        enable_privacy_logging=True,
        enable_circuit_breaker=True,
    )

    if args.health_check:
        status = guardrails.get_health_status()
        print("=== Health Status ===")
        print(json.dumps(status, indent=2, default=str))
        return

    if args.text:
        result = guardrails.process_request(args.text, user_id=args.user_id)
        print("=== Prediction Result ===")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Please provide text to classify with --text parameter")


if __name__ == "__main__":
    main()
