# src/spam_classifier/domain/entities.py
"""
Domain Entities - DDD Layer
Represents core business objects with identity
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from uuid import uuid4
from enum import Enum


class EmailLabel(Enum):
    """Email classification labels"""
    SPAM = "spam"
    HAM = "ham"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Supported ML model types"""
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"


@dataclass
class Email:
    """
    Email entity representing an email message
    Core domain object with identity
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    subject: str = ""
    body: str = ""
    sender: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    label: EmailLabel = EmailLabel.UNKNOWN
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity invariants"""
        if not self.body and not self.subject:
            raise ValueError("Email must have either subject or body")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def full_text(self) -> str:
        """Get combined subject and body text"""
        return f"{self.subject} {self.body}".strip()
    
    @property
    def is_classified(self) -> bool:
        """Check if email has been classified"""
        return self.label != EmailLabel.UNKNOWN
    
    def classify(self, label: EmailLabel, confidence: float) -> None:
        """Classify the email with a label and confidence score"""
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        self.label = label
        self.confidence = confidence
    
    def extract_features(self, feature_dict: Dict[str, float]) -> None:
        """Store extracted features for this email"""
        self.features = feature_dict.copy()


@dataclass
class ClassificationResult:
    """
    Value object representing classification outcome
    """
    email_id: str
    predicted_label: EmailLabel
    confidence: float
    probabilities: Dict[EmailLabel, float]
    model_used: ModelType
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        """Validate result invariants"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        # Validate probabilities sum to 1
        if self.probabilities:
            prob_sum = sum(self.probabilities.values())
            if not 0.99 <= prob_sum <= 1.01:  # Allow small floating point errors
                raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")
    
    @property
    def is_spam(self) -> bool:
        """Check if classified as spam"""
        return self.predicted_label == EmailLabel.SPAM
    
    @property
    def is_confident(self, threshold: float = 0.8) -> bool:
        """Check if prediction meets confidence threshold"""
        return self.confidence >= threshold


@dataclass
class MLModel:
    """
    ML Model entity representing a trained classification model
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    model_type: ModelType = ModelType.NAIVE_BAYES
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    trained_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    is_trained: bool = False
    training_samples: int = 0
    
    def mark_as_trained(self, samples: int, metrics: Dict[str, float]) -> None:
        """Mark model as trained with performance metrics"""
        self.is_trained = True
        self.trained_at = datetime.now()
        self.training_samples = samples
        self.metrics = metrics.copy()
    
    @property
    def accuracy(self) -> float:
        """Get model accuracy"""
        return self.metrics.get('accuracy', 0.0)
    
    @property
    def precision(self) -> float:
        """Get model precision"""
        return self.metrics.get('precision', 0.0)
    
    @property
    def recall(self) -> float:
        """Get model recall"""
        return self.metrics.get('recall', 0.0)
    
    @property
    def f1_score(self) -> float:
        """Get model F1 score"""
        return self.metrics.get('f1_score', 0.0)


@dataclass
class Dataset:
    """
    Dataset entity for managing training/test data
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    total_samples: int = 0
    spam_samples: int = 0
    ham_samples: int = 0
    split_ratio: Dict[str, float] = field(default_factory=lambda: {
        'train': 0.7, 'validation': 0.15, 'test': 0.15
    })
    is_balanced: bool = False
    preprocessing_applied: List[str] = field(default_factory=list)
    
    @property
    def spam_ratio(self) -> float:
        """Calculate spam to total ratio"""
        if self.total_samples == 0:
            return 0.0
        return self.spam_samples / self.total_samples
    
    @property
    def ham_ratio(self) -> float:
        """Calculate ham to total ratio"""
        if self.total_samples == 0:
            return 0.0
        return self.ham_samples / self.total_samples
    
    def validate_split(self) -> bool:
        """Validate that split ratios sum to 1"""
        total = sum(self.split_ratio.values())
        return 0.99 <= total <= 1.01


@dataclass
class TrainingSession:
    """
    Training session entity tracking model training process
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str = ""
    dataset_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    
    def start(self) -> None:
        """Start the training session"""
        self.status = "running"
        self.started_at = datetime.now()
    
    def complete(self, final_metrics: Dict[str, float]) -> None:
        """Complete the training session"""
        self.status = "completed"
        self.completed_at = datetime.now()
        self.metrics_history.append(final_metrics)
    
    def fail(self, error_message: str) -> None:
        """Mark session as failed"""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.logs.append(f"ERROR: {error_message}")
    
    def add_log(self, message: str) -> None:
        """Add log entry"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
    
    @property
    def duration_seconds(self) -> float:
        """Calculate training duration"""
        if not self.completed_at:
            return (datetime.now() - self.started_at).total_seconds()
        return (self.completed_at - self.started_at).total_seconds()