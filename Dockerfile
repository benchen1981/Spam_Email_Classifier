FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
COPY . .
RUN pip install -e .
EXPOSE 8501
CMD ["streamlit", "run", "src/spam_classifier/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]