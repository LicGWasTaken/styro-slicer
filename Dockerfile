# Start from the existing PyMesh image
FROM pymesh/pymesh

# Install the colorama package
RUN pip install colorama

# Copy your code into the container
COPY ./code /workspace
COPY ./obj /workspace

# Set the working directory
WORKDIR /workspace

# Optionally, run a script or start a shell
# CMD ["python", "test_pymesh.py"]
CMD ["/bin/bash"]