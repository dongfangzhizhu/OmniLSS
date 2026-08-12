# Phase 5.6: non-root runtime + pinned base image.
# NOTE: pin the base image digest after CI verification, e.g.
#   FROM python:3.11-slim@sha256:<digest>
# (docker not available in this workspace to resolve the digest).
FROM python:3.11-slim
WORKDIR /app

# Run as an unprivileged user (Phase 5.6 supply-chain hardening).
RUN useradd --create-home --uid 10001 appuser
USER appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
