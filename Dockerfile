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
RUN apt update && apt install -y libgomp1 libx11-6 libgl1 binutils libglib2.0-dev

# RUN pyinstaller --onefile --distpath ./exe/dist --workpath ./exe/build -i ./tgm.ico ./v2/main.py
CMD ["bash", "-c", "cd /workspace/code && exec /bin/bash"]
# CMD ["bash", "-c", "cd .."]

