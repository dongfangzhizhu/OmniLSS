FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e . pytest
CMD ["python", "-m", "pytest", "tests/test_integration.py", "-v"]
