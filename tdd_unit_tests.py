# tests/unit/test_domain.py
"""
TDD Unit Tests for Domain Entities
Following Test-Driven Development methodology
"""
import pytest
from datetime import datetime
from spam_classifier.domain.entities import (
    Email, EmailLabel, ClassificationResult, MLModel, 
    ModelType, Dataset, TrainingSession
)


class TestEmailEntity:
    """Test suite for Email entity following TDD"""
    
    def test_email_creation_with_valid_data(self):
        """Test: Should create email with valid data"""
        email = Email(
            subject="Test Subject",
            body="Test Body",
            sender="test@example.com"
        )
        
        assert email.id is not None
        assert email.subject == "Test Subject"
        assert email.body == "Test Body"
        assert email.label == EmailLabel.UNKNOWN
        assert email.confidence == 0.0
    
    def test_email_requires_content(self):
        """Test: Should raise error if no subject or body"""
        with pytest.raises(ValueError, match="Email must have either subject or body"):
            Email(subject="", body="", sender="test@example.com")
    
    def test_email_full_text_property(self):
        """Test: Should combine subject and body"""
        email = Email(subject="Hello", body="World")
        assert email.full_text == "Hello World"
    
    def test_email_classification(self):
        """Test: Should classify email with label and confidence"""
        email = Email(subject="Test", body="Content")
        email.classify(EmailLabel.SPAM, 0.95)
        
        assert email.label == EmailLabel.SPAM
        assert email.confidence == 0.95
        assert email.is_classified
    
    def test_email_classification_invalid_confidence(self):
        """Test: Should reject invalid confidence values"""
        email = Email(subject="Test", body="Content")
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            email.classify(EmailLabel.SPAM, 1.5)
    
    def test_email_feature_extraction(self):
        """Test: Should store extracted features"""
        email = Email(subject="Test", body="Content")
        features = {"word_count": 2, "special_chars": 0}
        email.extract_features(features)
        
        assert email.features == features
        assert "word_count" in email.features


class TestClassificationResult:
    """Test suite for ClassificationResult value object"""
    
    def test_result_creation_valid(self):
        """Test: Should create valid classification result"""
        result = ClassificationResult(
            email_id="test-123",
            predicted_label=EmailLabel.SPAM,
            confidence=0.92,
            probabilities={EmailLabel.SPAM: 0.92, EmailLabel.HAM: 0.08},
            model_used=ModelType.NAIVE_BAYES
        )
        
        assert result.email_id == "test-123"
        assert result.predicted_label == EmailLabel.SPAM
        assert result.is_spam
    
    def test_result_probabilities_must_sum_to_one(self):
        """Test: Should validate probabilities sum to 1"""
        with pytest.raises(ValueError, match="Probabilities must sum to 1"):
            ClassificationResult(
                email_id="test-123",
                predicted_label=EmailLabel.SPAM,
                confidence=0.92,
                probabilities={EmailLabel.SPAM: 0.5, EmailLabel.HAM: 0.3},
                model_used=ModelType.NAIVE_BAYES
            )
    
    def test_result_confidence_threshold(self):
        """Test: Should check confidence threshold"""
        result = ClassificationResult(
            email_id="test-123",
            predicted_label=EmailLabel.SPAM,
            confidence=0.85,
            probabilities={EmailLabel.SPAM: 0.85, EmailLabel.HAM: 0.15},
            model_used=ModelType.NAIVE_BAYES
        )
        
        assert result.is_confident(threshold=0.8)
        assert not result.is_confident(threshold=0.9)


class TestMLModelEntity:
    """Test suite for MLModel entity"""
    
    def test_model_creation(self):
        """Test: Should create ML model with default values"""
        model = MLModel(
            name="Spam Detector",
            model_type=ModelType.NAIVE_BAYES
        )
        
        assert model.id is not None
        assert model.name == "Spam Detector"
        assert not model.is_trained
        assert model.training_samples == 0
    
    def test_model_training_marking(self):
        """Test: Should mark model as trained with metrics"""
        model = MLModel(name="Test Model")
        metrics = {
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.97,
            'f1_score': 0.95
        }
        
        model.mark_as_trained(samples=1000, metrics=metrics)
        
        assert model.is_trained
        assert model.training_samples == 1000
        assert model.accuracy == 0.95
        assert model.trained_at is not None
    
    def test_model_metric_properties(self):
        """Test: Should access metric properties"""
        model = MLModel(name="Test Model")
        metrics = {
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.94,
            'f1_score': 0.92
        }
        model.mark_as_trained(1000, metrics)
        
        assert model.accuracy == 0.92
        assert model.precision == 0.90
        assert model.recall == 0.94
        assert model.f1_score == 0.92


class TestDatasetEntity:
    """Test suite for Dataset entity"""
    
    def test_dataset_creation(self):
        """Test: Should create dataset with valid data"""
        dataset = Dataset(
            name="Spam Dataset",
            source="Chapter03/datasets",
            total_samples=1000,
            spam_samples=400,
            ham_samples=600
        )
        
        assert dataset.name == "Spam Dataset"
        assert dataset.total_samples == 1000
        assert dataset.spam_ratio == 0.4
        assert dataset.ham_ratio == 0.6
    
    def test_dataset_split_validation(self):
        """Test: Should validate split ratios"""
        dataset = Dataset(
            name="Test Dataset",
            split_ratio={'train': 0.7, 'validation': 0.15, 'test': 0.15}
        )
        
        assert dataset.validate_split()
        
        # Invalid split
        dataset.split_ratio = {'train': 0.5, 'test': 0.3}
        assert not dataset.validate_split()
    
    def test_dataset_empty_samples(self):
        """Test: Should handle empty dataset"""
        dataset = Dataset(name="Empty", total_samples=0)
        
        assert dataset.spam_ratio == 0.0
        assert dataset.ham_ratio == 0.0


class TestTrainingSession:
    """Test suite for TrainingSession entity"""
    
    def test_session_creation(self):
        """Test: Should create training session"""
        session = TrainingSession(
            model_id="model-123",
            dataset_id="dataset-456"
        )
        
        assert session.id is not None
        assert session.status == "pending"
        assert session.model_id == "model-123"
    
    def test_session_lifecycle(self):
        """Test: Should manage session lifecycle"""
        session = TrainingSession(model_id="m1", dataset_id="d1")
        
        # Start session
        session.start()
        assert session.status == "running"
        
        # Complete session
        metrics = {'accuracy': 0.92}
        session.complete(metrics)
        assert session.status == "completed"
        assert session.completed_at is not None
        assert len(session.metrics_history) == 1
    
    def test_session_failure_handling(self):
        """Test: Should handle session failure"""
        session = TrainingSession(model_id="m1", dataset_id="d1")
        session.start()
        
        session.fail("Out of memory error")
        
        assert session.status == "failed"
        assert session.completed_at is not None
        assert any("ERROR" in log for log in session.logs)
    
    def test_session_logging(self):
        """Test: Should add log entries"""
        session = TrainingSession(model_id="m1", dataset_id="d1")
        
        session.add_log("Training started")
        session.add_log("Epoch 1 completed")
        
        assert len(session.logs) == 2
        assert "Training started" in session.logs[0]
    
    def test_session_duration_calculation(self):
        """Test: Should calculate duration"""
        session = TrainingSession(model_id="m1", dataset_id="d1")
        session.start()
        
        # Duration should be positive
        duration = session.duration_seconds
        assert duration >= 0


# Hypothesis-based property testing
from hypothesis import given, strategies as st

class TestPropertyBasedTesting:
    """Property-based tests using Hypothesis"""
    
    @given(
        subject=st.text(min_size=1, max_size=100),
        body=st.text(min_size=1, max_size=1000)
    )
    def test_email_full_text_always_contains_content(self, subject, body):
        """Property: full_text should always contain subject and body"""
        email = Email(subject=subject, body=body)
        full_text = email.full_text
        
        assert subject in full_text or body in full_text
    
    @given(confidence=st.floats(min_value=0.0, max_value=1.0))
    def test_valid_confidence_values_accepted(self, confidence):
        """Property: Valid confidence values should be accepted"""
        email = Email(subject="Test", body="Content")
        email.classify(EmailLabel.SPAM, confidence)
        
        assert email.confidence == confidence
    
    @given(
        spam_count=st.integers(min_value=0, max_value=10000),
        ham_count=st.integers(min_value=0, max_value=10000)
    )
    def test_dataset_ratios_always_sum_to_one(self, spam_count, ham_count):
        """Property: Spam and ham ratios should sum to 1"""
        total = spam_count + ham_count
        if total == 0:
            return  # Skip empty dataset
        
        dataset = Dataset(
            name="Test",
            total_samples=total,
            spam_samples=spam_count,
            ham_samples=ham_count
        )
        
        ratio_sum = dataset.spam_ratio + dataset.ham_ratio
        assert 0.99 <= ratio_sum <= 1.01  # Account for floating point


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=spam_classifier.domain"])