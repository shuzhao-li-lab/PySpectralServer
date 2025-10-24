# --- Stage 1: Builder ---
# Use a full Python image to build dependencies
FROM python:3.10-slim-bookworm as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install build tools just in case (ms_entropy might have C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Image ---
# Use a minimal base image
FROM python:3.10-slim-bookworm as final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/venv/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Create a non-root user for security
# -r: system user, -s /bin/false: no shell
RUN useradd -r -s /bin/false appuser

# Set the working directory
WORKDIR /app

# Copy the application code
COPY server.py .

# Create the directories for persistent data and give the non-root user ownership
# The script will write to these, so it needs permission.
RUN mkdir libraries && chown -R appuser:appuser libraries && \
    mkdir logs && chown -R appuser:appuser logs

# Switch to the non-root user
USER appuser

# Declare the volumes for persistent data (libraries and logs)
VOLUME /app/libraries
VOLUME /app/logs

# Expose the port the app runs on
EXPOSE 5000

# Run the application using Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "server:app"]