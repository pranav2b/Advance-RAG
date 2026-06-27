from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    cohere_api_key: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    collection_naive: str = "pg_naive"
    collection_contextual: str = "pg_contextual"

    class Config:
        env_file = ".env"


settings = Settings()
