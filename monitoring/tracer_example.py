from langchain_ollama import OllamaLLM
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from langchain_core.prompts import ChatPromptTemplate

# Service name is required for most backends
resource = Resource.create(attributes={
    SERVICE_NAME: "fede-llm",
    "service.version": "1.0.0",
    "deployment.environment": "development"
})


def configure_tracer():
    # Configure the OTLP exporter for your custom endpoint
    tracer_provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4319", insecure=True))
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)


configure_tracer()
tracer = trace.get_tracer("fede-llm-tracer")
with tracer.start_as_current_span("span-name") as span:
    # do some work that 'span' will track
    span_context = span.get_span_context()
    try:
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        llm = OllamaLLM(model="llama3.1:8b")
        chain = prompt | llm
        result = chain.invoke({"topic": "programming"})
    except Exception as e:
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise
