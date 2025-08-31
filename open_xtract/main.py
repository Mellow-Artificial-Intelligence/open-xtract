import requests
import json
import base64
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any, List
load_dotenv()

def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URL format."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        # Determine MIME type based on file extension
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
        elif image_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        else:
            mime_type = 'image/jpeg'  # default
        return f"data:{mime_type};base64,{encoded}"

def fetch_image_url_as_data_url(image_url: str) -> str:
    """Fetch a remote image by URL and return a base64 data URL."""
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get('Content-Type', '').lower()
    if content_type.startswith('image/'):
        mime_type = content_type.split(';')[0]
    else:
        # best-effort fallback based on URL
        if image_url.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_url.lower().endswith('.webp'):
            mime_type = 'image/webp'
        elif image_url.lower().endswith('.gif'):
            mime_type = 'image/gif'
        else:
            mime_type = 'image/jpeg'
    encoded = base64.b64encode(resp.content).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"

def detect_content_kind(content_url: str) -> str:
    """Return 'image' or 'document' based on URL/path/data URL; defaults to 'document'."""
    lower = content_url.lower()
    # Data URLs
    if lower.startswith('data:image/'):
        return 'image'
    if lower.startswith('data:application/pdf') or lower.startswith('data:text/'):
        return 'document'
    # Local path
    if os.path.exists(content_url):
        if any(lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']):
            return 'image'
        if any(lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt', '.md']):
            return 'document'
        return 'document'
    # URL/path by extension
    if any(lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']):
        return 'image'
    if any(lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt', '.md']):
        return 'document'
    return 'document'

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
}

class Extract:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = url,
        default_transforms: Optional[List[str]] = None
    ) -> None:
        self.base_url = base_url
        self.model = model
        if not self.model:
            raise ValueError("Model name is required. Pass model to Extract(model=...).")
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required. Set env var or pass api_key.")
        # Defaults to middle-out unless explicitly disabled at call site with []
        self.default_transforms = ["middle-out"] if default_transforms is None else default_transforms
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def extract(
        self,
        content_url: str,
        filename: str = "content.pdf",
        schema: Optional[Dict[str, Any]] = None,
        prompt: str = "Analyze this content",
        use_base64: bool = False,
        transforms: Optional[List[str]] = None
    ) -> Any:
        # Normalize transforms
        if transforms is None:
            transforms = self.default_transforms

        # Auto-detect content type
        content_type = detect_content_kind(content_url)

        # Build content array and plugins
        content_array: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]

        if content_type == 'image':
            image_url_to_send = content_url
            if content_url.startswith('data:image/'):
                use_base64 = True
            elif use_base64:
                if os.path.exists(content_url):
                    image_url_to_send = encode_image_to_base64(content_url)
                else:
                    image_url_to_send = fetch_image_url_as_data_url(content_url)
            content_array.append({
                "type": "image_url",
                "image_url": {"url": image_url_to_send}
            })
            plugins: List[Dict[str, Any]] = []
        else:
            content_array.append({
                "type": "file",
                "file": {"filename": filename, "file_data": content_url}
            })
            plugins = [{
                "id": "file-parser",
                "pdf": {"engine": "mistral-ocr"}
            }]

        messages = [{"role": "user", "content": content_array}]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "plugins": plugins
        }
        if transforms is not None:
            payload["transforms"] = transforms
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("name", "extraction"),
                    "strict": schema.get("strict", True),
                    "schema": schema.get("schema", schema)
                }
            }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if schema and "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
            return result
        # If image failed to fetch remotely, retry with base64
        if content_type == 'image' and not use_base64 and response.status_code == 400 and 'Failed to extract' in response.text:
            retry_image_url = encode_image_to_base64(content_url) if os.path.exists(content_url) else fetch_image_url_as_data_url(content_url)
            messages[0]["content"][1]["image_url"]["url"] = retry_image_url
            retry_payload: Dict[str, Any] = {"model": self.model, "messages": messages}
            if transforms is not None:
                retry_payload["transforms"] = transforms
            if schema:
                retry_payload["response_format"] = payload.get("response_format")
            retry_resp = requests.post(self.base_url, headers=self.headers, json=retry_payload)
            if retry_resp.status_code == 200:
                return retry_resp.json()
            raise Exception(f"API image base64 retry failed with status {retry_resp.status_code}: {retry_resp.text}")

        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

def extract_content_info(
    content_url: str,
    model: str,
    filename: str = "content.pdf",
    schema: Optional[Dict[str, Any]] = None,
    prompt: str = "Analyze this content",
    use_base64: bool = False,
    transforms: Optional[List[str]] = None
):
    """Backward-compatible wrapper that delegates to the Extract class."""
    extractor = Extract(model=model)
    return extractor.extract(
        content_url=content_url,
        filename=filename,
        schema=schema,
        prompt=prompt,
        use_base64=use_base64,
        transforms=transforms,
    )

# Example usage
if __name__ == "__main__":
    # Structured output schemas
    document_schema = {
        "name": "document_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "main_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of main points from the document"
                },
                "key_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key concepts discussed"
                }
            },
            "required": ["title", "main_points"],
            "additionalProperties": False
        }
    }

    image_schema = {
        "name": "image_description",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Overall description of the image"},
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key objects or entities present"
                },
                "dominant_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dominant colors observed"
                }
            },
            "required": ["description", "objects"],
            "additionalProperties": False
        }
    }

    # Example 1: Document processing (Structured Outputs)
    print("=== Document Processing (Structured Outputs) ===")
    extractor = Extract(model="google/gemini-2.5-flash-lite")
    doc_result = extractor.extract(
        content_url="https://bitcoin.org/bitcoin.pdf",
        schema=document_schema,
        prompt="Extract the title, main points, and key concepts."
    )
    print("Document result:", json.dumps(doc_result, indent=2))

    # Example 2: Document processing (Structured Outputs, alternate prompt)
    print("\n=== Document Processing (Structured Outputs, alternate prompt) ===")
    structured_doc_result = extractor.extract(
        content_url="https://bitcoin.org/bitcoin.pdf",
        schema=document_schema,
        prompt="Summarize the document and list the key concepts."
    )
    print("Structured document result:", json.dumps(structured_doc_result, indent=2))

    # Example 3: Image processing with URL (Structured Outputs)
    print("\n=== Image Processing (Structured Outputs) ===")
    image_result = extractor.extract(
        content_url="https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png",
        schema=image_schema,
        prompt="Describe this image, list objects and dominant colors."
    )
    print("Image result:", json.dumps(image_result, indent=2))

    # Example 4: Image processing with base64 (Structured Outputs) (uncomment and provide local image path)
    # print("\n=== Base64 Image Processing (Structured Outputs) ===")
    # base64_data_url = encode_image_to_base64("path/to/your/image.jpg")
    # base64_image_result = extractor.extract(
    #     content_url=base64_data_url,
    #     schema=image_schema,
    #     prompt="Describe this image, list objects and dominant colors."
    # )
    # print("Base64 image result:", json.dumps(base64_image_result, indent=2))

    # Example 5: Auto-detection of content type (Structured Outputs)
    print("\n=== Auto-detection (Structured Outputs) ===")
    auto_result = extractor.extract(
        content_url="https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png",
        schema=image_schema,
        prompt="Analyze this content"
    )
    print("Auto-detected result:", json.dumps(auto_result, indent=2))
