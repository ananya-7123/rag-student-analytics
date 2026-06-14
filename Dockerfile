FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /workspace

# Copy the requirements file from the root to install dependencies
COPY requirements.txt .

# Install dependencies cleanly without caching unnecessary data
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else from your root folder into the container
COPY . .

# FastAPI default port
EXPOSE 8000

# Run your existing development API script to start the server
CMD ["python", "run_dev_api.py"]