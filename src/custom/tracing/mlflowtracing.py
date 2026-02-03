import mlflow
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any


class MLflowGenAITracer:
    """
    MLflow-based tracer for GenAI/LLM applications.

    Provides lightweight utilities to:
    - Manage MLflow runs
    - Create spans for tracing execution steps
    - Log LLM and embedding calls
    - Record custom metrics
    """

    def __init__(self, experiment_name: str = "genai-experiments"):
        """
        Initialize the tracer and set the MLflow experiment.

        Args:
            experiment_name: Name of the MLflow experiment where runs
                             and traces will be recorded.
        """
        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def _ensure_run(self):
        """
        Ensure there is an active MLflow run.
        Starts a default run if none exists.
        """
        if mlflow.active_run() is None:
            self.active_run = mlflow.start_run(run_name="auto_run")

    def start_trace(self, run_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Start a new MLflow run for tracing.

        Args:
            run_name: Name of the MLflow run.
            metadata: Optional key-value parameters to log with the run.
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        if metadata:
            mlflow.log_params(metadata)

    def end_trace(self):
        """
        End the currently active MLflow run.
        """
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing a logical execution step.

        Args:
            name: Span name describing the operation.
            attributes: Optional attributes to attach to the span.

        Example:
            with tracer.span("retrieve_documents"):
                ...
        """
        span = mlflow.start_span(name)
        start_time = time.time()

        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        try:
            yield
        finally:
            span.set_attribute("duration_sec", time.time() - start_time)
            span.end()

    def log_llm_call(
        self,
        model: str,
        prompt: Any,
        response: Any,
        usage: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an LLM generation call as a span.

        Args:
            model: Model name used for generation.
            prompt: Input prompt sent to the model.
            response: Model response output.
            usage: Optional token or cost usage statistics.
        """
        with mlflow.start_span("llm_generation") as span:
            span.set_attribute("model", model)
            span.set_attribute("prompt", str(prompt))
            span.set_attribute("response", str(response))

            if usage:
                for k, v in usage.items():
                    span.set_attribute(f"usage_{k}", v)

    def log_embedding_call(self, model: str, vector_size: int):
        """
        Log an embedding generation call.

        Args:
            model: Embedding model name.
            vector_size: Size (dimension) of the generated embedding.
        """
        with mlflow.start_span("embedding_generation") as span:
            span.set_attribute("model", model)
            span.set_attribute("vector_size", vector_size)


    def log_metric(self, name: str, value: float):
        """
        Log a numeric metric to the active MLflow run.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self._ensure_run()
        mlflow.log_metric(name, value)

    def score_trace(
        self,
        name: str,
        value: float,
        comment: Optional[str] = None,
        scorer: Optional[str] = None,
    ):
        """
        Log an evaluation score for the entire trace/run.

        Args:
            name: Name of the score (e.g., relevance, faithfulness)
            value: Score value
            comment: Optional explanation
            scorer: Model or method used to compute the score
        """
        self._ensure_run()

        # Score value → metric (for charts & comparison)
        mlflow.log_metric(name, value)

        # Extra context → tags
        if comment:
            mlflow.set_tag(f"{name}_comment", comment)
        if scorer:
            mlflow.set_tag(f"{name}_scorer", scorer)

