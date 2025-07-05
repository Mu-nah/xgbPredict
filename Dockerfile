FROM python:3.10-slim

WORKDIR /app

# Install dependencies directly (no requirements.txt)
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    ta \
    xgboost \
    vaderSentiment \
    scikit-learn \
    python-dotenv

COPY . .

CMD ["python", "xgb_predictor.py"]
