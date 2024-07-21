# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock first to leverage Docker cache
COPY UI\ Only/Pipfile UI\ Only/Pipfile.lock /app/

# Install pipenv and project dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the .env file to the container
COPY .env /app/.env

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "UI Only/main.py"]
