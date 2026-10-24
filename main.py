"""ASGI entry point: `uvicorn main:app`."""

try:  # use the OS certificate store (needed behind TLS-inspecting proxies; harmless elsewhere)
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

from engine.api.app import create_app

app = create_app()
