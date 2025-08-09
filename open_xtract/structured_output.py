from pydantic import BaseModel, Field
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import io
import base64
import httpx


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _read_bytes_from_source(source: str, timeout: float = 30.0) -> bytes:
    if _is_url(source):
        response = httpx.get(source, timeout=timeout)
        response.raise_for_status()
        return response.content
    with open(source, "rb") as file:
        return file.read()

def parse_pdf(pdf_path: str, model: str = "gpt-5-nano", max_tokens_image_description: int = 1024, output_format: BaseModel = None) -> list[Document]:
    """
    Parse a PDF file and return a list of documents.
    """
    loader = PyPDFLoader(
        pdf_path,
        mode="page",
        images_inner_format="markdown-img",
        images_parser=LLMImageBlobParser(model=ChatOpenAI(model=model, max_tokens=max_tokens_image_description)),
    )
    docs = loader.load()
    model = ChatOpenAI(model=model)
    model = model.with_structured_output(output_format)
    result = model.invoke(docs)
    return result

def parse_image(img_path: str, model: str = "gpt-5-nano", output_format: BaseModel = None) -> BaseModel | str:
    model = ChatOpenAI(model=model)
    image_prompt = """
    You are an assistant tasked with summarizing images for retrieval.
    1. These summaries will be embedded and used to retrieve the raw image.
    Give a concise summary of the image that is well optimized for retrieval
    2. extract all the text from the image.
    Do not exclude any content from the page.
    """
    message_content = [
        {
            "type": "text",
            "text": image_prompt,
        }
    ]

    if _is_url(img_path):
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": img_path},
            }
        )
        
    else:
        image_bytes = io.BytesIO()
        with open(img_path, "rb") as img_file:
            image_bytes.write(img_file.read())
        img_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }
        )
    if output_format:
        model = model.with_structured_output(output_format)
    result = model.invoke([HumanMessage(content=message_content)])
    
    return result

def parse_text(text: str, model: str = "gpt-5-nano", output_format: BaseModel = None) -> BaseModel | str:
    model = ChatOpenAI(model=model)
    model = model.with_structured_output(output_format)
    result = model.invoke(text)
    return result

def parse_video(
    video_path: str,
    model: str = "gemini-2.0-flash",
    mime_type: str = "video/mp4",
    prompt: str = "Describe the first few frames of the video.",
    output_format: BaseModel | None = None,
) -> BaseModel | str:
    llm = ChatGoogleGenerativeAI(model=model)
    if output_format:
        llm = llm.with_structured_output(output_format)

    # Support local files and URLs by normalizing to bytes and base64
    video_bytes = _read_bytes_from_source(video_path)
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "media", "data": encoded_video, "mime_type": mime_type},
        ]
    )

    result = llm.invoke([message])
    return result

if __name__ == "__main__":
    class PDFSummary(BaseModel):
        summary: str = Field(description="A concise summary of the PDF")
        text: str = Field(description="The text from the PDF")
    docs = parse_pdf("/Users/colesmcintosh/Projects/open-xtract/test_docs/ian_lease.pdf", output_format=PDFSummary)
    print(docs)
    image_path = "test_docs/ghibli-port.png"
    class ImageSummary(BaseModel):
        summary: str = Field(description="A concise summary of the image that is well optimized for retrieval")
        text: str = Field(description="The text from the image")
    image_prompt = parse_image(image_path, output_format=ImageSummary)
    print(image_prompt)

    video_path = "test_docs/ghibli-port.mp4"
    class VideoSummary(BaseModel):
        summary: str = Field(description="A concise summary of the video")
        text: str = Field(description="The text from the video")
    video_prompt = parse_video(video_path, output_format=VideoSummary)
    print(video_prompt)