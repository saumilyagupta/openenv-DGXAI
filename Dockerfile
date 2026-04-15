FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the Layer B KB corpus at image build time so the server boots fast.
RUN python3 -m groundloop.skills_scraper || echo "scraper build skipped (no user skills in image)"

EXPOSE 7860
CMD ["uvicorn", "groundloop_env.app:app", "--host", "0.0.0.0", "--port", "7860"]
