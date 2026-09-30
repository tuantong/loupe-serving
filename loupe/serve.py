import argparse
import os
import secrets
import threading
from collections.abc import Callable

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from loupe.api import ContextBudgetExceeded, DecideRequest, DecideResponse, Policy, RequestRejected
from loupe.calibration import Calibration
from loupe.engine import Engine

BACKENDS = ("fake", "mlx", "hf", "graph", "vllm")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _backend(backend_name: str, env: dict[str, str]):
    if backend_name == "fake":
        from loupe.backend_fake import FakeBackend

        return FakeBackend()
    if backend_name == "mlx":
        from loupe.backend_mlx import MLXBackend

        return MLXBackend.from_env(env)
    if backend_name == "hf":
        from loupe.backend_hf import HFBackend

        backend = HFBackend.from_env()
        window = float(env.get("LOUPE_BATCH_WINDOW_MS", "0"))
        if window > 0:
            from loupe.batcher import BatchedBackend

            backend = BatchedBackend(backend, window_ms=window, max_batch=backend.max_batch)
        return backend
    if backend_name == "graph":
        from loupe.backend_graph import GraphBackend

        backend = GraphBackend.from_env(env)
        backend.warm()
        return backend
    if backend_name == "vllm":
        from loupe.backend_vllm import VLLMBackend

        return VLLMBackend.from_env()
    raise ValueError(f"unknown backend {backend_name!r}, expected one of {BACKENDS}")


def build_engine(backend_name: str, env: dict[str, str] | None = None) -> Engine:
    env = env or os.environ
    settings = {}
    if "LOUPE_CALIBRATION" in env:
        settings["calibration"] = Calibration.load(env["LOUPE_CALIBRATION"])
        if settings["calibration"].model_name:
            settings["model_name"] = settings["calibration"].model_name
    if "LOUPE_CONTEXT_BUDGET" in env:
        settings["context_budget"] = int(env["LOUPE_CONTEXT_BUDGET"])
    if "LOUPE_MODEL_NAME" in env:
        settings["model_name"] = env["LOUPE_MODEL_NAME"]
    if "LOUPE_MAX_THINKING_TOKENS" in env:
        settings["max_thinking_tokens"] = int(env["LOUPE_MAX_THINKING_TOKENS"])
    if "LOUPE_SCRATCH_MIN_TOKENS" in env:
        settings["scratch_min_tokens"] = int(env["LOUPE_SCRATCH_MIN_TOKENS"])
    if "LOUPE_MAX_SCRATCH_TOKENS" in env:
        settings["max_scratch_tokens"] = int(env["LOUPE_MAX_SCRATCH_TOKENS"])
    return Engine(_backend(backend_name, env), **settings)


class _EngineLoader:
    """Builds the engine in the background. A load failure is kept in `error`, so the server reports it
    instead of answering "loading" forever."""

    def __init__(self, backend_name: str):
        self.engine: Engine | None = None
        self.error: str | None = None
        threading.Thread(target=self._load, args=(backend_name,), daemon=True).start()

    def _load(self, backend_name: str) -> None:
        try:
            self.engine = build_engine(backend_name)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def __call__(self) -> Engine | None:
        return self.engine


def start_loading(backend_name: str) -> Callable[[], Engine | None]:
    return _EngineLoader(backend_name)


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def create_app(
    engine_provider: Callable[[], Engine | None],
    token: str | None,
    max_body_bytes: int = 1_000_000,
    default_policy: Policy | None = None,
    serial_free: bool = False,
) -> FastAPI:
    hidden = {} if token is None else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="loupe", **hidden)
    queue = threading.Lock() if not serial_free else _NoLock()

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return
        if not secrets.compare_digest((authorization or "").encode(), f"Bearer {token}".encode()):
            raise HTTPException(
                status_code=401,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.middleware("http")
    async def cap_body(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None and int(declared) > max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        path = ".".join(str(p) for p in first["loc"] if p != "body")
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "validation", "path": path, "message": first["msg"]}},
        )

    @app.exception_handler(RequestRejected)
    async def rejected(_: Request, exc: RequestRejected) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "validation", "path": exc.path, "message": exc.message}},
        )

    @app.get("/healthz")
    def healthz():
        if engine_provider() is not None:
            return {"status": "ok"}
        error = getattr(engine_provider, "error", None)
        if error:
            return JSONResponse(status_code=503, content={"status": "failed", "error": error})
        return {"status": "loading"}

    @app.post("/v1/systemone", response_model=DecideResponse, dependencies=[Depends(authorize)])
    def systemone(body: DecideRequest) -> DecideResponse:
        engine = engine_provider()
        if engine is None:
            error = getattr(engine_provider, "error", None)
            raise HTTPException(status_code=503, detail=f"model failed to load: {error}" if error else "model loading")
        if default_policy is not None and "policy" not in body.model_fields_set:
            body = body.model_copy(update={"policy": default_policy})
        try:
            with queue:
                return engine.decide(body)
        except ContextBudgetExceeded as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="loupe.serve")
    parser.add_argument("--backend", choices=BACKENDS, default=os.environ.get("LOUPE_BACKEND", "fake"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    token = os.environ.get("LOUPE_API_TOKEN")
    if args.host not in LOOPBACK and token is None:
        parser.error("refusing to bind a non-loopback host without LOUPE_API_TOKEN")
    max_body_bytes = int(os.environ.get("LOUPE_MAX_BODY_BYTES", 1_000_000))
    default_policy = Policy.model_validate_json(os.environ["LOUPE_POLICY"]) if os.environ.get("LOUPE_POLICY") else None
    serial_free = args.backend == "hf" and float(os.environ.get("LOUPE_BATCH_WINDOW_MS", "0")) > 0
    app = create_app(start_loading(args.backend), token, max_body_bytes=max_body_bytes, default_policy=default_policy, serial_free=serial_free)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
