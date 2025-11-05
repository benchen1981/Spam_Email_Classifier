from setuptools import setup, find_packages

setup(
    name="spam-email-classifier",
    version="1.0.0",
    description="Professional Spam Email Classifier with AI/ML",
    author="Ben Chen",
    author_email="benchen1981@github.com",
    url="https://github.com/benchen1981/Spam_Email_Classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)