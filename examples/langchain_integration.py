"""Example showing integration with LangChain ecosystem."""

from pydantic import BaseModel
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from open_xtract import PDFProcessor


class ResearchPaper(BaseModel):
    """Schema for research paper metadata."""
    title: str
    authors: list[str]
    abstract: str
    methodology: str = ""
    conclusions: str = ""


def process_with_text_splitting():
    """Example showing how to use extracted documents with LangChain splitters."""
    print("=== Processing with Text Splitting ===")
    
    # Initialize processor
    processor = PDFProcessor(
        llm_provider="openai",
        model="gpt-4o-mini"
    )
    
    # Process PDF
    result = processor.process_pdf(
        pdf_path="research_paper.pdf",
        schema=ResearchPaper
    )
    
    if result.success:
        print(f"Extracted metadata:")
        print(f"  Title: {result.data.title}")
        print(f"  Authors: {', '.join(result.data.authors)}")
        
        # The markdown files contain the full extracted content
        # We can load and split them for vector storage
        if result.markdown_files:
            # Read the extracted content
            all_content = []
            for md_file in result.markdown_files:
                with open(md_file, 'r') as f:
                    content = f.read()
                    # Create a Document object for LangChain
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(md_file)}
                    )
                    all_content.append(doc)
            
            # Use LangChain splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            splits = text_splitter.split_documents(all_content)
            print(f"\nCreated {len(splits)} chunks for vector storage")
            
            # Show first chunk
            if splits:
                print(f"\nFirst chunk preview:")
                print(splits[0].page_content[:200] + "...")


def custom_document_processing():
    """Example showing how to access the raw Document objects."""
    print("\n=== Custom Document Processing ===")
    
    # For advanced users who want direct access to Document objects
    from open_xtract.workflows.nodes.pdf_processing import process_pdf_node
    from open_xtract.schemas.types import ProcessingState
    from open_xtract.models.providers import create_provider
    from open_xtract.models.base import LLMConfig
    import asyncio
    
    async def custom_process():
        # Create provider
        config = LLMConfig(
            provider="openai",
            api_key="your-key",
            model="gpt-4o"
        )
        provider = create_provider(config)
        
        # Create initial state
        state = ProcessingState(
            pdf_path="document.pdf",
            schema={"properties": {"text": {"type": "string"}}},
            supports_vision=provider.supports_vision()
        )
        
        # Process PDF directly
        state = await process_pdf_node(state, provider)
        
        # Access Document objects directly
        for doc in state.documents:
            print(f"Page {doc.metadata.get('page', 'unknown')}:")
            print(f"  Extraction method: {doc.metadata.get('extraction_method')}")
            print(f"  Content length: {len(doc.page_content)} chars")
            
            # You can now use these Document objects with any LangChain component
            # e.g., embeddings, vector stores, chains, etc.
        
        return state.documents
    
    # Run the async function
    # documents = asyncio.run(custom_process())


def integration_with_vector_store():
    """Example showing how to integrate with vector stores."""
    print("\n=== Vector Store Integration ===")
    
    # This is a conceptual example - you'll need to install vector store dependencies
    processor = PDFProcessor(model="gpt-4o-mini")
    
    result = processor.process_pdf(
        pdf_path="document.pdf",
        schema={
            "properties": {
                "document_type": {"type": "string"},
                "key_topics": {"type": "array", "items": {"type": "string"}}
            }
        }
    )
    
    if result.success:
        print(f"Document type: {result.data.get('document_type')}")
        print(f"Key topics: {result.data.get('key_topics', [])}")
        
        # Example of how you would add to a vector store
        print("\nTo add to a vector store:")
        print("1. Load the markdown files as Documents")
        print("2. Split documents into chunks")
        print("3. Generate embeddings")
        print("4. Store in your preferred vector database")
        print("\nExample with Chroma:")
        print("""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load documents from markdown files
documents = []
for md_file in result.markdown_files:
    with open(md_file, 'r') as f:
        doc = Document(
            page_content=f.read(),
            metadata={"source": str(md_file), **result.data}
        )
        documents.append(doc)

# Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
""")


if __name__ == "__main__":
    process_with_text_splitting()
    custom_document_processing()
    integration_with_vector_store()