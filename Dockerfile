# https://docs.docker.com/engine/reference/builder/

FROM ghcr.io/astral-sh/uv:python3.12-bookworm
COPY dist/*.whl .
RUN uv pip install --system *.whl
CMD ["cs231n-spr-2025", "--help"]

COPY ./src /app
ENV PYTHONPATH="${PYTHONPATH}:/app"
WORKDIR /app

# CMD ["cs231n-spr-2025", "--help"]
# Example command
CMD ["uvicorn", "example_app.main:app", "--host", "0.0.0.0", "--reload"]
