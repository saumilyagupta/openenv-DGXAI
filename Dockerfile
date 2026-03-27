# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src directory and the baseline inference script into the container
COPY src/ ./src/
COPY inference.py .

# Specify the command to run on container start
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
