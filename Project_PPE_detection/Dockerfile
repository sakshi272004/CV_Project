# Use the official Python image as a base
FROM python:3.9-slim

# Set environment variables to avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libgl1 && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir opencv-python-headless==4.5.5.64
RUN pip install --no-cache-dir ultralytics==8.3.49

# Ensure the output directory exists
RUN mkdir -p /app/output

# Set the command to run the application
CMD ["python", "PPE_Detection.py"]
