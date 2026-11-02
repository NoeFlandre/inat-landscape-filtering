# CPU image. The CLI is the entry point: docker run IMAGE --help
# Behind a TLS-inspecting proxy, pass its CA bundle: docker build --secret id=ca,src=CA.crt .
FROM public.ecr.aws/docker/library/python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=secret,id=ca \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --offline
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /work
ENTRYPOINT ["inat-landscape", "--workdir", "/work"]
