FROM python:3.9-slim

# Set the working directory
WORKDIR /myapp

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Expose the port the app runs on
EXPOSE 8888

# Command to run the application
CMD ["python", "app.py"]