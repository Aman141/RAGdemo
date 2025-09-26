from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

	openai_api_key: str | None = None
	default_embedding_model: str = "text-embedding-3-small"


settings = Settings()  # Evaluate once and import where needed


