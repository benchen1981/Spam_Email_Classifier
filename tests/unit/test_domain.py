"""Unit Tests for Domain Entities (TDD)"""
import pytest
from spam_classifier.domain.entities import Email, EmailLabel

def test_email_creation():
    email = Email(subject="Test", body="Content")
    assert email.subject == "Test"
    assert email.body == "Content"
    assert email.label == EmailLabel.UNKNOWN

def test_email_requires_content():
    with pytest.raises(ValueError):
        Email(subject="", body="")

def test_email_full_text():
    email = Email(subject="Hello", body="World")
    assert email.full_text == "Hello World"

def test_email_classification():
    email = Email(subject="Test", body="Content")
    email.label = EmailLabel.SPAM
    email.confidence = 0.95
    assert email.is_classified
    assert email.confidence == 0.95