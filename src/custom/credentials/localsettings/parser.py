from pydantic_settings import BaseSettings, SettingsConfigDict


class ArxivParserConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARXIV_Parser__",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )
    pdf_dir: str = "./datas/arxiv_pdfs"
    max_pages:int = 25
    max_file_size_mb: int = 30
    max_concurrency: int = 5
    do_ocr: bool = False
    do_table_structure: bool = True

