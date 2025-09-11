FROM python:3.10-slim

# Install system dependencies for OpenCV and libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the source code into the container
COPY . /app

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
