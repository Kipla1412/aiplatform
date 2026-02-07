import os
import logging
from typing import Optional
from dotenv import load_dotenv

from src.custom.connectors.factory import ConnectorFactory
from src.custom.credentials.factory import CredentialFactory
from src.custom.loaders.opensearch import OpenSearchService
from src.custom.embeddings.jarxivembeddings import JinaEmbeddingsService
from src.custom.llm.ollama import OllamaClient
from src.custom.llm.cache.redis import CacheClient
from src.custom.tracing.tracker import RAGTracer
from src.custom.tracing.langfuseengine import LangfuseTracer
from src.custom.agent.agentrag import AgenticRAGService
from src.custom.agent.config.arxivconfig import GraphConfig
# from src.custom.observability.ragtracer import RAGTracer
# from src.custom.observability.langfusetracer import LangfuseTracer

load_dotenv()
logger = logging.getLogger(__name__)


class ServiceContainer:
    def __init__(self):
        self.embeddings: Optional[JinaEmbeddingsService] = None
        self.search: Optional[OpenSearchService] = None
        self.llm: Optional[OllamaClient] = None
        self.cache: Optional[CacheClient] = None
        self.tracer: Optional[RAGTracer] = None
        self.agent: Optional[AgenticRAGService] = None


    def init_all(self):
        self.init_embeddings()
        self.init_search()
        self.init_llm()
        self.init_cache()
        self.init_tracing()
        self.init_agent() 

    def init_embeddings(self):
      
        creds = CredentialFactory.get_provider("jina").get_credentials()
        connector = ConnectorFactory.get_connector("jina", creds)
        self.embeddings = JinaEmbeddingsService(connector, creds)

    def init_search(self):
    
        creds = CredentialFactory.get_provider("opensearch").get_credentials()
        connector = ConnectorFactory.get_connector("opensearch", creds)
        self.search = OpenSearchService(connector, creds)

    def init_llm(self):
        creds = CredentialFactory.get_provider("ollama").get_credentials()
        connector = ConnectorFactory.get_connector("ollama", creds)
        self.llm = OllamaClient(connector, creds)

    def init_cache(self):
        try:
            creds = CredentialFactory.get_provider("redis").get_credentials()
            connector = ConnectorFactory.get_connector("redis", creds)
            self.cache = CacheClient(connector, creds)
        except Exception as e:
            logger.warning(f"Redis disabled: {e}")

    def init_tracing(self):
        try:
            creds = CredentialFactory.get_provider("langfuse").get_credentials()
            connector = ConnectorFactory.get_connector("langfuse", creds)
            self.tracer = RAGTracer(LangfuseTracer(connector))
        except Exception as e:
            logger.warning(f"Langfuse disabled: {e}")

    def init_agent(self):
        if not (self.search and self.llm and self.embeddings):
            raise RuntimeError("Core services must be initialized before agent")

        graph_config = GraphConfig()  # or load from YAML later

        self.agent = AgenticRAGService(
            opensearch_client=self.search,
            ollama_client=self.llm,
            embeddings_client=self.embeddings,
            langfuse_tracer=self.tracer.tracer if self.tracer else None,
            graph_config=graph_config,
            cache_client=self.cache,
        )

services = ServiceContainer()
