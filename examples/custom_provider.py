"""Example using a custom OpenAI-compatible provider."""

from pydantic import BaseModel

from open_xtract import PDFProcessor


class ContractData(BaseModel):
    """Schema for contract data extraction."""
    contract_type: str
    parties: list[str]
    effective_date: str
    expiration_date: str = ""
    key_terms: list[str] = []
    governing_law: str = ""


def main():
    # Example 1: Using Ollama (local LLM)
    processor_ollama = PDFProcessor(
        llm_provider="ollama",
        api_key="not-needed",  # Ollama doesn't require API key
        model="llama2",        # or any model you have in Ollama
        base_url="http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
    )
    
    # Example 2: Using Azure OpenAI
    processor_azure = PDFProcessor(
        llm_provider="azure",
        api_key="your-azure-api-key",
        model="gpt-4",
        base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
        api_version="2023-05-15"  # Azure-specific parameter
    )
    
    # Example 3: Using any OpenAI-compatible API
    processor_custom = PDFProcessor(
        llm_provider="custom",
        api_key="your-api-key",
        model="your-model-name",
        base_url="https://your-api-endpoint.com/v1"
    )
    
    # Process a contract PDF
    result = processor_ollama.process_pdf(
        pdf_path="contract.pdf",
        schema=ContractData
    )
    
    if result.success:
        print("✓ Successfully extracted contract data!")
        print(f"\nContract Type: {result.data.contract_type}")
        print(f"Parties: {', '.join(result.data.parties)}")
        print(f"Effective Date: {result.data.effective_date}")
        
        if result.data.expiration_date:
            print(f"Expiration Date: {result.data.expiration_date}")
        
        if result.data.key_terms:
            print("\nKey Terms:")
            for term in result.data.key_terms:
                print(f"  - {term}")
        
        if result.data.governing_law:
            print(f"\nGoverning Law: {result.data.governing_law}")
    else:
        print(f"✗ Extraction failed: {result.error}")


if __name__ == "__main__":
    main()