"""Example using JSON schema instead of Pydantic model."""

import json

from open_xtract import PDFProcessor


# Define JSON schema for a research paper
paper_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Title of the paper"
        },
        "authors": {
            "type": "array",
            "description": "List of author names",
            "items": {"type": "string"}
        },
        "abstract": {
            "type": "string",
            "description": "Paper abstract or summary"
        },
        "keywords": {
            "type": "array",
            "description": "Keywords or tags",
            "items": {"type": "string"}
        },
        "publication_date": {
            "type": "string",
            "description": "Publication date"
        }
    },
    "required": ["title", "authors", "abstract"]
}


def main():
    # Initialize processor
    processor = PDFProcessor(
        llm_provider="openai",
        model="gpt-4o-mini"
    )
    
    # Process PDF with JSON schema
    result = processor.process_pdf(
        pdf_path="research_paper.pdf",
        schema=paper_schema
    )
    
    if result.success:
        print("✓ Successfully extracted paper metadata!")
        print(f"\nTitle: {result.data['title']}")
        
        authors = result.data.get('authors', [])
        if authors:
            print(f"Authors: {', '.join(authors)}")
        
        abstract = result.data.get('abstract', '')
        if abstract:
            print(f"\nAbstract: {abstract[:200]}...")
        
        keywords = result.data.get('keywords', [])
        if keywords:
            print(f"\nKeywords: {', '.join(keywords)}")
        
        # Save as JSON
        with open('paper_metadata.json', 'w') as f:
            json.dump(result.data, f, indent=2)
        print("\nMetadata saved to paper_metadata.json")
    else:
        print(f"✗ Extraction failed: {result.error}")


if __name__ == "__main__":
    main()