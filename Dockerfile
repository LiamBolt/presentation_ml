# 1. Pick a slim Python base
FROM python:3.11-slim

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy only requirements first (to leverage Docker cache)
COPY requirements.txt .

# 4. Install OS‑level deps (if you need any, e.g. gcc) then Python libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your entire app code into /app
COPY . .

# 6. Expose the default Streamlit port
EXPOSE 8501

# 7. Launch Streamlit pointing at your entry script
ENTRYPOINT ["streamlit", "run", "presentation_ml/stroke_risk_prediction/app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
