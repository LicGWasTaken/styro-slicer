# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /workspace

# Copy your code into the container
# COPY ./code /workspace/code
COPY ./v2 /workspace/code
COPY ./obj /workspace/obj
COPY settings.json /workspace/
COPY requirements.txt /workspace/

# Upgrade pip
RUN pip install --upgrade pip

# Install dependences
RUN pip install --no-cache-dir -r requirements.txt
# Dependeces for open3d
RUN apt update && apt install -y libgomp1 libx11-6 libgl1

# Optionally, run a script or start a shell
# CMD ["python", "test_pymesh.py"]
# CMD ["/bin/bash"]
CMD ["bash", "-c", "cd /workspace/code && exec /bin/bash"]

