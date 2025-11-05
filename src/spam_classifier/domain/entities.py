"""Domain Entities - Core business objects"""
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

class EmailLabel(Enum):
    SPAM = "spam"
    HAM = "ham"
    UNKNOWN = "unknown"

class ModelType(Enum):
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"

@dataclass
class Email:
    id: str = field(default_factory=lambda: str(uuid4()))
    subject: str = ""
    body: str = ""
    sender: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    label: EmailLabel = EmailLabel.UNKNOWN
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.body and not self.subject:
            raise ValueError("Email must have either subject or body")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def full_text(self) -> str:
        return f"{self.subject} {self.body}".strip()
    
    @property
    def is_classified(self) -> bool:
        return self.label != EmailLabel.UNKNOWN

@dataclass
class ClassificationResult:
    email_id: str
    predicted_label: EmailLabel
    confidence: float
    probabilities: dict[EmailLabel, float]
    model_used: ModelType
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if not abs(sum(self.probabilities.values()) - 1.0) < 1e-6:
            raise ValueError("Probabilities must sum to 1")
    
    @property
    def is_spam(self) -> bool:
        return self.predicted_label == EmailLabel.SPAM
    
    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold

@dataclass
class MLModel:
    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    model_type: ModelType = ModelType.NAIVE_BAYES
    is_trained: bool = False
    training_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    trained_at: Optional[datetime] = None

    def mark_as_trained(self, samples: int, metrics: Dict[str, float]):
        self.is_trained = True
        self.training_samples = samples
        self.accuracy = metrics.get('accuracy', 0.0)
        self.precision = metrics.get('precision', 0.0)
        self.recall = metrics.get('recall', 0.0)
        self.f1_score = metrics.get('f1_score', 0.0)
        self.trained_at = datetime.now()

@dataclass
class Dataset:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str
    source: str = ""
    total_samples: int = 0
    spam_samples: int = 0
    ham_samples: int = 0
    split_ratio: Dict[str, float] = field(default_factory=lambda: {'train': 0.7, 'validation': 0.15, 'test': 0.15})

    @property
    def spam_ratio(self) -> float:
        return self.spam_samples / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def ham_ratio(self) -> float:
        return self.ham_samples / self.total_samples if self.total_samples > 0 else 0.0

    def validate_split(self) -> bool:
        return abs(sum(self.split_ratio.values()) - 1.0) < 1e-6

@dataclass
class TrainingSession:
    id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str
    dataset_id: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def start(self):
        self.status = "running"
        self.start_time = datetime.now()
        self.add_log("Training session started")

    def complete(self, metrics: Dict[str, Any]):
        self.status = "completed"
        self.completed_at = datetime.now()
        self.metrics_history.append(metrics)
        self.add_log("Training session completed")

    def fail(self, error_message: str):
        self.status = "failed"
        self.completed_at = datetime.now()
        self.add_log(f"ERROR: {error_message}")

    def add_log(self, message: str):
        self.logs.append(f"{datetime.now()}: {message}")

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.completed_at:
            return (self.completed_at - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0