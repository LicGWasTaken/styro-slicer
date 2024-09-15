# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /workspace

# Copy your code into the container
COPY ./code /workspace/code
COPY ./obj /workspace/obj
COPY ./plots /workspace/plots
COPY ./output /workspace/output
COPY requirements.txt /workspace/

# Upgrade pip
RUN pip install --upgrade pip

# Install dependences
RUN pip install --no-cache-dir -r requirements.txt

# Optionally, run a script or start a shell
# CMD ["python", "test_pymesh.py"]
CMD ["/bin/bash"]

