import logging

from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs._internal.export import SimpleLogRecordProcessor
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry._logs import set_logger_provider, get_logger
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# Service name is required for most backends
resource = Resource.create(attributes={
    SERVICE_NAME: "fede_llm",
    "service.version": "1.0.0",
    "deployment.environment": "development"
})

provider = LoggerProvider()
console_processor = BatchLogRecordProcessor(ConsoleLogExporter())
otel_processor = SimpleLogRecordProcessor(OTLPLogExporter(endpoint="http://localhost:4319"))
provider.add_log_record_processor(console_processor)
provider.add_log_record_processor(otel_processor)
# Sets the global default logger provider
set_logger_provider(provider)

logger = get_logger(__name__)

handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
logging.basicConfig(handlers=[handler], level=logging.INFO)

logging.info("This is an OpenTelemetry log record!")
