import logging
from pathlib import Path

from src.custom.extractors.pdfparserservice import PDFParserService
from src.custom.credentials.localsettings.parser import ArxivParserConfig
import asyncio

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def main():
    config = ArxivParserConfig()
    config_dict = {"docling": config.model_dump()}

    output_dir = Path("./data/output_json")
    output_dir.mkdir(parents=True, exist_ok=True)

    service = PDFParserService(config_dict)
    print("Starting PDF Parsing...")
    all_content = await service.parse_all_pdfs()

    for i, doc in enumerate(all_content):

            print("\n" + "="*50)

            source_name = doc.metadata.get('source', f'document_{i}')
            # If source is 'paper.pdf', this makes it 'paper'
            file_stem = Path(source_name).stem 
            json_path = output_dir / f"{file_stem}.json"

            print(f"PROCESSING: {source_name}")
            print(f"DOCUMENT: {doc.metadata.get('source', 'Unknown')}")
            print(f"SECTIONS FOUND: {len(doc.sections)}")

            import json
            with open(json_path, "w", encoding="utf-8") as f:
                # .model_dump() converts Pydantic object to dictionary
                json.dump(doc.model_dump(), f, indent=4, ensure_ascii=False)
        
            print(f"SAVED TO: {json_path}")
            if doc.sections:
                first = doc.sections[2]
                print(f"FIRST SECTION TITLE: {first.title}")
                print(f"SNIPPET: {first.content[:200]}...")
            print("="*50)

if __name__ == "__main__":
    asyncio.run(main())