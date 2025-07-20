from langchain_ollama import OllamaLLM
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from langchain_core.prompts import ChatPromptTemplate

# Service name is required for most backends
resource = Resource.create(attributes={
    SERVICE_NAME: "fede-llm"
})

# Configure the OTLP exporter for your custom endpoint
tracerProvider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4319", insecure=True))
tracerProvider.add_span_processor(processor)
trace.set_tracer_provider(tracerProvider)

# reader = PeriodicExportingMetricReader(
#     OTLPMetricExporter(endpoint="<traces-endpoint>/v1/metrics")
# )
# meterProvider = MeterProvider(resource=resource, metric_readers=[reader])
# metrics.set_meter_provider(meterProvider)

tracer = trace.get_tracer("test")

with tracer.start_as_current_span("span-name") as span:
    # do some work that 'span' will track
    print("doing some work...")
    # When the 'with' block goes out of scope, 'span' is closed for you

# Create and run a LangChain application
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = OllamaLLM(model="llama3.1:8b")
chain = prompt | llm

result = chain.invoke({"topic": "programming"})
print(result)
