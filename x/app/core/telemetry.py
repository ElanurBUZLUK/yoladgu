Dosya: backend/app/core/telemetry.py
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

try:
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except Exception:
    RedisInstrumentor = None

OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")

class Telemetry:
    @staticmethod
    def init(app, engine=None):
        resource = Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "edu-backend")})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT}/v1/traces"))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()
        if engine:
            SQLAlchemyInstrumentor().instrument(engine=engine)
        if RedisInstrumentor:
            try:
                RedisInstrumentor().instrument()
            except Exception:
                pass
