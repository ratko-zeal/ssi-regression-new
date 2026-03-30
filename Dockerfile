# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files from your repo into the container
COPY . .

# Install dependencies from BOTH files
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r streamlit_app/requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8080

# Command to run the app
CMD ["python", "-m", "streamlit", "run", "streamlit_app/Home.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.enableWebsocketCompression=false"]
