"""
Spam Email Classifier - Professional Software Engineering Project
Following CRISP-DM, TDD, SDD, BDD, DDD methodologies

Project Structure:
spam_email_classifier/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── CRISP_DM_Process.md
│   ├── architecture.md
│   └── api_documentation.md
├── src/
│   └── spam_classifier/
│       ├── __init__.py
│       ├── domain/                    # DDD: Domain Layer
│       │   ├── __init__.py
│       │   ├── entities.py           # Email, Classification entities
│       │   ├── value_objects.py      # EmailAddress, ClassificationResult
│       │   ├── repositories.py       # Abstract repositories
│       │   └── services.py           # Domain services
│       ├── application/               # DDD: Application Layer
│       │   ├── __init__.py
│       │   ├── use_cases.py          # Business use cases
│       │   └── dto.py                # Data Transfer Objects
│       ├── infrastructure/            # DDD: Infrastructure Layer
│       │   ├── __init__.py
│       │   ├── data_access.py        # Repository implementations
│       │   ├── ml_models.py          # ML model implementations
│       │   └── persistence.py        # Data persistence
│       ├── data_science/              # CRISP-DM Pipeline
│       │   ├── __init__.py
│       │   ├── business_understanding.py
│       │   ├── data_understanding.py
│       │   ├── data_preparation.py
│       │   ├── modeling.py
│       │   ├── evaluation.py
│       │   └── deployment.py
│       └── web/                       # Streamlit Application
│           ├── __init__.py
│           ├── app.py
│           ├── components.py
│           └── visualizations.py
├── tests/                             # TDD & BDD Tests
│   ├── __init__.py
│   ├── unit/                          # TDD Unit Tests
│   │   ├── test_domain.py
│   │   ├── test_services.py
│   │   └── test_ml_models.py
│   ├── integration/                   # Integration Tests
│   │   ├── test_use_cases.py
│   │   └── test_pipeline.py
│   ├── bdd/                           # BDD Feature Tests
│   │   ├── features/
│   │   │   ├── email_classification.feature
│   │   │   └── model_training.feature
│   │   └── steps/
│   │       └── classification_steps.py
│   └── conftest.py                    # Pytest configuration
├── data/
│   ├── raw/                           # Chapter03/datasets
│   ├── processed/
│   └── models/
├── notebooks/                         # Exploratory Data Analysis
│   └── 01_eda_crisp_dm.ipynb
└── deployment/
    ├── Dockerfile
    └── docker-compose.yml

"""

# requirements.txt content
REQUIREMENTS = """
# Core Dependencies
python>=3.9
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Machine Learning & NLP
nltk>=3.8.0
spacy>=3.6.0
transformers>=4.30.0
torch>=2.0.0
gensim>=4.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
wordcloud>=1.9.0

# Web Application
streamlit>=1.28.0
streamlit-aggrid>=0.3.4
streamlit-option-menu>=0.3.6

# Testing (TDD/BDD)
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-bdd>=6.1.0
hypothesis>=6.82.0
behave>=1.2.6

# Code Quality
black>=23.7.0
flake8>=6.0.0
mypy>=1.4.0
pylint>=2.17.0
isort>=5.12.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=1.3.0

# Data Processing
joblib>=1.3.0
pickle5>=0.0.12

# Logging & Monitoring
loguru>=0.7.0
mlflow>=2.5.0

# Configuration
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0
"""

# setup.py content
SETUP_PY = """
from setuptools import setup, find_packages

setup(
    name="spam-email-classifier",
    version="1.0.0",
    description="Professional Spam Email Classifier using AI/ML with Software Engineering Best Practices",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.14.0",
        "nltk>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-bdd>=6.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
"""

print("Project structure and configuration files defined.")
print("\nNext steps:")
print("1. Create the directory structure")
print("2. Implement domain entities (DDD)")
print("3. Write tests first (TDD)")
print("4. Implement CRISP-DM pipeline")
print("5. Create BDD feature specifications")
print("6. Build Streamlit visualization app")