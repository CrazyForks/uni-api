FROM python:3.11 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --output-file /tmp/requirements.txt && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

FROM python:3.11-slim-bullseye
ARG SOURCE_COMMIT=unknown
ENV SOURCE_COMMIT=${SOURCE_COMMIT}
EXPOSE 8000
WORKDIR /home
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
ENTRYPOINT ["python", "main.py"]
