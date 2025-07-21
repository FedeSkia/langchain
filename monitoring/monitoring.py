from langchain_ollama import OllamaLLM
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs._internal.export import BatchLogRecordProcessor

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from langchain_core.prompts import ChatPromptTemplate
import logging_example

# Service name is required for most backends
resource = Resource.create(attributes={
    SERVICE_NAME: "fede-llm",
    "service.version": "1.0.0",
    "deployment.environment": "development"

})


def configure_tracer():
    # Configure the OTLP exporter for your custom endpoint
    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4319", insecure=True))
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)


def configure_logger() -> logging.Logger:
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    log_processor = BatchLogRecordProcessor(
        OTLPLogExporter(endpoint="http://localhost:4319")
    )
    logger_provider.add_log_record_processor(log_processor)
    # Setup logging handler
    otel_handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    fede_llm_logger = logging.getLogger("fede-llm")
    fede_llm_logger.addHandler(otel_handler)
    fede_llm_logger.addHandler(logging.StreamHandler())
    # Configure standard Python logging
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     handlers=[
    #         logging.StreamHandler(),  # Console
    #         otel_handler  # OpenTelemetry -> Loki
    #     ]
    # )
    return fede_llm_logger


configure_tracer()
logger = configure_logger()
tracer = trace.get_tracer("fede-llm-tracer")
with tracer.start_as_current_span("span-name") as span:
    # do some work that 'span' will track
    span_context = span.get_span_context()

    logger.info("LLM processing started", extra={
        "span_id": hex(span_context.span_id),
        "trace_id": hex(span_context.trace_id),
        "operation": "llm_invoke",
        "model": "llama3.1:8b"
    })
    try:
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        llm = OllamaLLM(model="llama3.1:8b")
        chain = prompt | llm
        logger.info("Invoking chain", extra={
            "topic": "programming",
            "span_id": hex(span_context.span_id),
            "trace_id": hex(span_context.trace_id)
        })
        result = chain.invoke({"topic": "programming"})
        logger.info("LLM processing completed successfully.", extra={
            "status": "success",
            "result_length": len(result),
            "span_id": hex(span_context.span_id),
            "trace_id": hex(span_context.trace_id)
        })
    except Exception as e:
        logger.error("LLM processing failed", extra={
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "span_id": hex(span_context.span_id),
            "trace_id": hex(span_context.trace_id)
        })
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise
