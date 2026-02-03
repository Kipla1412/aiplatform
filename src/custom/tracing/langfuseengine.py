# # src/custom/tracing/langfuseengine.py

# import logging
# from contextlib import contextmanager
# from typing import Optional, Dict, Any
# from langfuse import get_client
# from langfuse import observe


# from langfuse import observe
# from src.custom.connectors.langfuse import LangfuseConnector

# logger = logging.getLogger(__name__)


# class LangfuseTracer:
#     """
#     Langfuse-based tracer built on the v3 `observe()` API.

#     Provides structured tracing for:
#     - Incoming requests
#     - Internal processing spans
#     - LLM generations
#     - Evaluation scores

#     Gracefully degrades when Langfuse is disabled.
#     """

#     def __init__(self, connector: LangfuseConnector):
#         """
#         Initialize the tracer with a Langfuse connector.

#         Args:
#             connector: Connector responsible for creating and managing
#                        the Langfuse client instance.
#         """
#         self.connector = connector
#         self.client = connector.get_client()

#         if self.client:
#             logger.info("LangfuseTracer initialized (v3 observe API)")
#         else:
#             logger.warning("Langfuse disabled — running without tracing")

#     @contextmanager
#     def trace_request(
#         self,
#         name: str,
#         input_data: Dict[str, Any],
#         user_id: Optional[str] = None,
#         session_id: Optional[str] = None,
#         metadata: Optional[Dict[str, Any]] = None,
#     ):
#         """
#         Create a top-level trace for a user request.

#         Args:
#             name: Name of the trace (e.g., endpoint or workflow name).
#             input_data: Structured input payload for observability.
#             user_id: Optional user identifier.
#             session_id: Optional session identifier.
#             metadata: Additional metadata to attach to the trace.

#         Yields:
#             Langfuse trace object, or None if tracing is disabled/fails.
#         """
#         if not self.client:
#             yield None
#             return

#         try:
#             with observe(name=name) as trace:
#                 # Attach structured data AFTER creating trace
#                 trace.update_current_trace(
#                     input=input_data,
#                     user_id=user_id,
#                     session_id=session_id,
#                 )
#                 yield trace
#         except Exception as e:
#             logger.error(f"Langfuse trace error: {e}")
#             yield None

#     @contextmanager
#     def span(self, name: str, input_data: Optional[Dict[str, Any]] = None):
#         """
#         Create a nested span inside the current trace.

#         Args:
#             name: Span name describing the operation.
#             input_data: Optional structured input for the span.

#         Yields:
#             Langfuse span object, or None if tracing is disabled/fails.
#         """
#         if not self.client:
#             yield None
#             return

#         try:
#             with observe(name=name) as span:
#                 if input_data:
#                     span.update(input=input_data)
#                 yield span
#         except Exception as e:
#             logger.error(f"Langfuse span error ({name}): {e}")
#             yield None

#     @contextmanager
#     def generation(self, model: str, prompt: str):
#         """
#         Create a span for an LLM generation event.

#         Args:
#             model: Name of the LLM used.
#             prompt: Prompt text sent to the model.

#         Yields:
#             Langfuse generation span, or None if tracing is disabled/fails.
#         """
#         if not self.client:
#             yield None
#             return

#         try:
#             with observe(name="llm_generation") as gen:
#                 gen.update_current_observation(
#                     model=model,
#                     input={"prompt": prompt},
#                 )
#                 yield gen
#         except Exception as e:
#             logger.error(f"Langfuse generation error: {e}")
#             yield None

#     def score(self, name: str, value: float, comment: Optional[str] = None):
#         """
#         Attach an evaluation score to the current trace.

#         Args:
#             name: Score name (e.g., "relevance", "accuracy").
#             value: Numeric score value.
#             comment: Optional descriptive comment.
#         """
#         if not self.client:
#             return

#         try:
#             langfuse_context.score_current_trace(
#                 name=name,
#                 value=value,
#                 comment=comment,
#             )
#         except Exception as e:
#             logger.error(f"Langfuse score error: {e}")

#     def shutdown(self):
#         """
#         Flush pending traces and safely shut down the Langfuse client.
#         """
#         try:
#             self.connector.shutdown()
#         except Exception as e:
#             logger.error(f"Langfuse shutdown error: {e}")


import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from src.custom.connectors.langfuse import LangfuseConnector

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse tracing client."""

    def __init__(self, connector: LangfuseConnector):
        
        self.connector = connector
        self.client: Optional[Langfuse] = connector.get_client()

        if self.client:
            logger.info("LangfuseTracer connected to Langfuse")
        else:
            logger.warning("LangfuseTracer running without Langfuse client")
        
    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing a RAG request.

        Args:
            query: The user's query
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata to attach to the trace

        Yields:
            Trace object if Langfuse is enabled, None otherwise
        """
        if not self.client:
            yield None
            return
        
        try:
            # Create a trace using v2 API
            trace = self.client.trace(
                name="rag_request",
                input={"query": query},
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
            )
            yield trace
        except Exception as e:
            logger.error(f"Error creating Langfuse trace: {e}")
            yield None

        finally:
            try:
                self.connector.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse after trace: {e}")

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span within a trace.

        Args:
            trace: Parent trace object
            name: Name of the span
            input_data: Input data for the span
            metadata: Additional metadata

        Returns:
            Span object if successful, None otherwise
        """
        if not trace or not self.client:
            return None

        try:
            # Create a span using v2 API
            return self.client.span(
                trace_id=trace.trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Error creating span {name}: {e}")
            return None

    def create_generation(
        self,
        trace,
        name: str,
        model: str,
        input_data: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a generation (LLM call) within a trace.

        Args:
            trace: Parent trace object
            name: Name of the generation
            model: Model name
            input_data: Input/prompt data
            output: Generated output
            metadata: Additional metadata
            usage: Token usage information

        Returns:
            Generation object if successful, None otherwise
        """
        if not trace or not self.client:
            return None

        try:
            # Create a generation using v2 API
            return self.client.generation(
                trace_id=trace.trace_id,
                name=name,
                model=model,
                input=input_data,
                output=output,
                metadata=metadata or {},
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Error creating generation {name}: {e}")
            return None

    def score_trace(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """
        Add a score to a trace.

        Args:
            trace: Trace object
            name: Score name (e.g., "relevance", "accuracy")
            value: Score value
            comment: Optional comment
        """
        if not trace or not self.client:
            return

        try:
            # Create a score using v2 API
            self.client.score(
                trace_id=trace.trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.error(f"Error scoring trace: {e}")

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """
        Update a span with output or additional metadata.

        Args:
            span: Span object to update
            output: Output data
            metadata: Additional metadata
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            status_message: Status message
        """
        if not span:
            return

        try:
            # For v2 API, we can update spans with end_time and output
            if output is not None:
                # Update the span with output data
                span.update(output=output)
            if metadata:
                span.update(metadata=metadata)
            if level:
                span.update(level=level)
            if status_message:
                span.update(status_message=status_message)
        except Exception as e:
            logger.error(f"Error updating span: {e}")

    def end_span(self, span, output: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        End a span with optional final output and metadata.

        Args:
            span: Span object to end
            output: Final output data
            metadata: Final metadata
        """
        if not span:
            return

        try:
            # Update with final data if provided
            if output is not None or metadata is not None:
                self.update_span(span, output=output, metadata=metadata)

            # End the span to capture proper timing
            span.end()
        except Exception as e:
            logger.error(f"Error ending span: {e}")

    def flush(self):
        """Flush any pending traces."""
        try:
            self.connector.flush()
        except Exception as e:
            logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Shutdown the Langfuse client."""
        try:
            self.connector.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down Langfuse: {e}")
