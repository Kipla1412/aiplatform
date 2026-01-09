from .rdbms import RDBMSExtractor
from .arxivparser import ArxivParser
class ExtractorFactory:
    @staticmethod
    def get_extractor(extractor_type: str,connection: str = None, config: dict = None):
        if extractor_type == "rdbms":
            return RDBMSExtractor(connection=connection, config=config)

        # elif extractor_type == "gmail":
        #     return GmailExtractor(connection=connection, config=config)
        
        elif extractor_type == "arxivparser":
            return PDFParserService(config=config)

        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")