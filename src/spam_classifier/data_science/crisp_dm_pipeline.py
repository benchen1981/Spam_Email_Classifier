# src/spam_classifier/data_science/crisp_dm_pipeline.py
"""
CRISP-DM (Cross-Industry Standard Process for Data Mining) Implementation
Six phases: Business Understanding, Data Understanding, Data Preparation,
Modeling, Evaluation, Deployment
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CRISPDMConfig:
    """Configuration for CRISP-DM pipeline"""
    data_path: str = "data/raw/emails.csv"
    model_save_path: str = "data/models/"
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    cross_validation_folds: int = 5


class Phase1_BusinessUnderstanding:
    """
    CRISP-DM Phase 1: Business Understanding
    Define objectives and requirements from business perspective
    """
    
    def __init__(self):
        self.business_objectives = {
            'primary': 'Classify emails as spam or ham with high accuracy',
            'secondary': [
                'Minimize false positives (legitimate emails marked as spam)',
                'Provide confidence scores for classifications',
                'Enable real-time email processing',
                'Support model retraining with new data'
            ]
        }
        self.success_criteria = {
            'accuracy': 0.90,
            'precision': 0.85,
            'recall': 0.90,
            'f1_score': 0.87,
            'false_positive_rate': 0.05
        }
        self.constraints = {
            'response_time_ms': 100,
            'model_size_mb': 50,
            'minimum_training_samples': 1000
        }
    
    def define_business_requirements(self) -> Dict[str, Any]:
        """Define and document business requirements"""
        logger.info("Phase 1: Business Understanding - Defining requirements")
        
        requirements = {
            'objectives': self.business_objectives,
            'success_criteria': self.success_criteria,
            'constraints': self.constraints,
            'stakeholders': ['Security Team', 'IT Department', 'End Users'],
            'risks': [
                'High false positive rate causing user frustration',
                'Model drift over time with evolving spam techniques',
                'Privacy concerns with email content processing'
            ]
        }
        
        logger.info(f"Business objectives defined: {self.business_objectives['primary']}")
        return requirements


class Phase2_DataUnderstanding:
    """
    CRISP-DM Phase 2: Data Understanding
    Initial data collection and exploration
    """
    
    def __init__(self, config: CRISPDMConfig):
        self.config = config
        self.data = None
        self.data_profile = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load email dataset from CSV"""
        logger.info(f"Phase 2: Loading data from {self.config.data_path}")
        
        try:
            self.data = pd.read_csv(self.config.data_path)
            logger.info(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self) -> Dict[str, Any]:
        """Explore and profile the dataset"""
        logger.info("Phase 2: Exploring data characteristics")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.data_profile = {
            'total_samples': len(self.data),
            'features': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
        }
        
        # Analyze label distribution if 'label' column exists
        if 'label' in self.data.columns:
            label_dist = self.data['label'].value_counts().to_dict()
            self.data_profile['label_distribution'] = label_dist
            self.data_profile['class_balance_ratio'] = (
                label_dist.get('spam', 0) / label_dist.get('ham', 1) 
                if 'ham' in label_dist else 0
            )
        
        # Text length statistics
        if 'text' in self.data.columns or 'body' in self.data.columns:
            text_col = 'text' if 'text' in self.data.columns else 'body'
            self.data[f'{text_col}_length'] = self.data[text_col].str.len()
            self.data_profile['text_length_stats'] = {
                'mean': self.data[f'{text_col}_length'].mean(),
                'median': self.data[f'{text_col}_length'].median(),
                'std': self.data[f'{text_col}_length'].std(),
                'min': self.data[f'{text_col}_length'].min(),
                'max': self.data[f'{text_col}_length'].max()
            }
        
        logger.info(f"Data exploration complete: {self.data_profile}")
        return self.data_profile
    
    def identify_data_quality_issues(self) -> List[str]:
        """Identify data quality problems"""
        issues = []
        
        if self.data_profile['duplicate_rows'] > 0:
            issues.append(f"Found {self.data_profile['duplicate_rows']} duplicate rows")
        
        for col, missing_count in self.data_profile['missing_values'].items():
            if missing_count > 0:
                issues.append(f"Column '{col}' has {missing_count} missing values")
        
        if 'class_balance_ratio' in self.data_profile:
            ratio = self.data_profile['class_balance_ratio']
            if ratio < 0.3 or ratio > 3.0:
                issues.append(f"Significant class imbalance detected: ratio = {ratio:.2f}")
        
        logger.info(f"Identified {len(issues)} data quality issues")
        return issues


class Phase3_DataPreparation:
    """
    CRISP-DM Phase 3: Data Preparation
    Clean, transform, and engineer features
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize text and apply lemmatization"""
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def prepare_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Complete data preparation pipeline"""
        logger.info("Phase 3: Data Preparation started")
        
        df = df.copy()
        
        # Handle missing values
        df[text_column] = df[text_column].fillna('')
        
        # Remove duplicates
        original_size = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {original_size - len(df)} duplicate rows")
        
        # Clean text
        logger.info("Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Tokenize (optional, for analysis)
        logger.info("Tokenizing text...")
        df['tokens'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Feature engineering
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['tokens'].apply(len)
        df['avg_word_length'] = df['cleaned_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        
        logger.info("Phase 3: Data Preparation complete")
        return df
    
    def create_feature_matrix(
        self, 
        texts: List[str], 
        fit: bool = True,
        config: CRISPDMConfig = None
    ) -> np.ndarray:
        """Create TF-IDF feature matrix"""
        logger.info("Creating TF-IDF feature matrix...")
        
        if config is None:
            config = CRISPDMConfig()
        
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
            X = self.vectorizer.fit_transform(texts)
            logger.info(f"Feature matrix created: {X.shape}")
        else:
            X = self.vectorizer.transform(texts)
            logger.info(f"Feature matrix transformed: {X.shape}")
        
        return X


class Phase4_Modeling:
    """
    CRISP-DM Phase 4: Modeling
    Select and apply various modeling techniques
    """
    
    def __init__(self, config: CRISPDMConfig):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def define_models(self) -> Dict[str, Any]:
        """Define multiple models to compare"""
        self.models = {
            'naive_bayes': MultinomialNB(alpha=0.1),
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.config.random_state,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='linear',
                probability=True,
                random_state=self.config.random_state
            )
        }
        
        logger.info(f"Defined {len(self.models)} models for comparison")
        return self.models
    
    def train_model(
        self, 
        model_name: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Any:
        """Train a specific model"""
        logger.info(f"Phase 4: Training {model_name} model...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        logger.info(f"{model_name} training complete")
        return model
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Train all defined models"""
        logger.info("Training all models...")
        
        trained_models = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        logger.info("All models trained successfully")
        return trained_models
    
    def cross_validate_model(
        self, 
        model_name: str, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info(f"Cross-validating {model_name}...")
        
        model = self.models[model_name]
        scores = cross_val_score(
            model, X, y,
            cv=self.config.cross_validation_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"{model_name} CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        return cv_results


class Phase5_Evaluation:
    """
    CRISP-DM Phase 5: Evaluation
    Evaluate model performance and validate against business objectives
    """
    
    def __init__(self, business_criteria: Dict[str, float]):
        self.business_criteria = business_criteria
        self.evaluation_results = {}
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logger.info(f"Phase 5: Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', pos_label='spam'),
            'recall': recall_score(y_test, y_pred, average='binary', pos_label='spam'),
            'f1_score': f1_score(y_test, y_pred, average='binary', pos_label='spam'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(
                (y_test == 'spam').astype(int), 
                y_prob
            )
            metrics['roc_curve'] = {
                'fpr': roc_curve((y_test == 'spam').astype(int), y_prob)[0].tolist(),
                'tpr': roc_curve((y_test == 'spam').astype(int), y_prob)[1].tolist()
            }
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True
        )
        
        self.evaluation_results[model_name] = metrics
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def check_business_criteria(self, metrics: Dict[str, float]) -> bool:
        """Validate metrics against business success criteria"""
        logger.info("Checking against business criteria...")
        
        meets_criteria = True
        for criterion, threshold in self.business_criteria.items():
            if criterion in metrics:
                actual = metrics[criterion]
                meets = actual >= threshold
                meets_criteria = meets_criteria and meets
                
                status = "✓ PASS" if meets else "✗ FAIL"
                logger.info(f"{criterion}: {actual:.4f} vs {threshold:.4f} {status}")
        
        return meets_criteria
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models"""
        logger.info("Comparing model performance...")
        
        comparison_data = []
        for model_name, metrics in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', None)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        logger.info("\n" + df_comparison.to_string())
        return df_comparison


class Phase6_Deployment:
    """
    CRISP-DM Phase 6: Deployment
    Deploy model to production and create monitoring plan
    """
    
    def __init__(self, model_save_path: str):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, vectorizer: Any, model_name: str) -> str:
        """Save trained model and vectorizer"""
        logger.info(f"Phase 6: Deploying {model_name}...")
        
        model_file = self.model_save_path / f"{model_name}_model.pkl"
        vectorizer_file = self.model_save_path / f"{model_name}_vectorizer.pkl"
        
        joblib.dump(model, model_file)
        joblib.dump(vectorizer, vectorizer_file)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Vectorizer saved to {vectorizer_file}")
        
        return str(model_file)
    
    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load trained model and vectorizer"""
        logger.info(f"Loading {model_name}...")
        
        model_file = self.model_save_path / f"{model_name}_model.pkl"
        vectorizer_file = self.model_save_path / f"{model_name}_vectorizer.pkl"
        
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
        
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    
    def create_monitoring_plan(self) -> Dict[str, Any]:
        """Create deployment monitoring plan"""
        monitoring_plan = {
            'metrics_to_track': [
                'accuracy', 'precision', 'recall', 'f1_score',
                'false_positive_rate', 'false_negative_rate',
                'average_confidence_score', 'prediction_latency'
            ],
            'monitoring_frequency': 'daily',
            'alert_thresholds': {
                'accuracy_drop': 0.05,  # Alert if accuracy drops by 5%
                'false_positive_increase': 0.10,
                'prediction_latency_ms': 200
            },
            'retraining_triggers': [
                'Accuracy drops below 0.85',
                'False positive rate exceeds 0.10',
                'New spam patterns detected',
                'Quarterly scheduled retraining'
            ],
            'data_drift_detection': {
                'method': 'KS-test',
                'features_to_monitor': ['text_length', 'word_count'],
                'threshold': 0.05
            }
        }
        
        logger.info("Deployment monitoring plan created")
        return monitoring_plan


# Complete CRISP-DM Pipeline Orchestrator
class CRISPDMPipeline:
    """Complete CRISP-DM pipeline orchestrator"""
    
    def __init__(self, config: CRISPDMConfig = None):
        self.config = config or CRISPDMConfig()
        self.phase1 = Phase1_BusinessUnderstanding()
        self.phase2 = Phase2_DataUnderstanding(self.config)
        self.phase3 = Phase3_DataPreparation()
        self.phase4 = Phase4_Modeling(self.config)
        self.phase5 = Phase5_Evaluation(self.phase1.success_criteria)
        self.phase6 = Phase6_Deployment(self.config.model_save_path)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute complete CRISP-DM pipeline"""
        logger.info("="*60)
        logger.info("CRISP-DM Pipeline Execution Started")
        logger.info("="*60)
        
        results = {}
        
        # Phase 1: Business Understanding
        results['business_requirements'] = self.phase1.define_business_requirements()
        
        # Phase 2: Data Understanding
        self.phase2.load_data()
        results['data_profile'] = self.phase2.explore_data()
        results['data_quality_issues'] = self.phase2.identify_data_quality_issues()
        
        # Phase 3: Data Preparation
        prepared_data = self.phase3.prepare_dataset(self.phase2.data)
        
        # Create feature matrix and split data
        X = self.phase3.create_feature_matrix(
            prepared_data['cleaned_text'].tolist(), 
            fit=True,
            config=self.config
        )
        y = prepared_data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        results['data_split'] = {
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'n_features': X_train.shape[1]
        }
        
        # Phase 4: Modeling
        self.phase4.define_models()
        self.phase4.train_all_models(X_train, y_train)
        
        # Phase 5: Evaluation
        for model_name, model in self.phase4.models.items():
            self.phase5.evaluate_model(model, X_test, y_test, model_name)
        
        results['model_comparison'] = self.phase5.compare_models()
        
        # Select best model
        best_model_name = results['model_comparison'].iloc[0]['Model']
        best_model = self.phase4.models[best_model_name]
        best_metrics = self.phase5.evaluation_results[best_model_name]
        
        results['best_model'] = {
            'name': best_model_name,
            'metrics': best_metrics,
            'meets_criteria': self.phase5.check_business_criteria(best_metrics)
        }
        
        # Phase 6: Deployment
        model_path = self.phase6.save_model(
            best_model, 
            self.phase3.vectorizer, 
            best_model_name
        )
        results['deployment'] = {
            'model_path': model_path,
            'monitoring_plan': self.phase6.create_monitoring_plan()
        }
        
        logger.info("="*60)
        logger.info("CRISP-DM Pipeline Execution Completed")
        logger.info("="*60)
        
        return results