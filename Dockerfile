FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir --no-build-isolation causal-conv1d
ENV LOUPE_HF_MODEL=Qwen/Qwen3.5-4B LOUPE_PROMPT_STYLE=compact LOUPE_GRAPH_MAX_TOKENS=1024
EXPOSE 8000
# LOUPE_API_TOKEN must be set at run time: the server does not listen beyond loopback without it
CMD ["python", "-m", "loupe.serve", "--backend", "graph", "--host", "0.0.0.0", "--port", "8000"]
