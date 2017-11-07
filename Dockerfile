# Use an official Python runtime as a parent image
FROM python:3.5.4

# Set the working directory to /app
WORKDIR /chatbot

# Copy the current directory contents into the container at /app
ADD . /chatbot

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Runs chatbot in interactive mode when the container launches
CMD ["python", "single_dialog.py", "--task_id", "7", "--train", "false", "--interactive", "true"]
