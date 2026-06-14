FROM python:3.11-slim

# Set working directory for the frontend service
WORKDIR /app

# Copy the frontend dependencies file
COPY requirements.txt .

# Install Streamlit and its UI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend app code (app.py) into the container
COPY . .

# Streamlit default port
EXPOSE 8501

# Command to launch Streamlit and ensure it broadcasts outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]