from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from open_xtract.main import Extract

app = FastAPI(title="OpenXtract API", version="0.1.0")


class ExtractionSchema(BaseModel):
    # Expecting the same shape we pass to Extract.extract (name, strict, schema)
    model_config = ConfigDict(populate_by_name=True)
    name: Optional[str] = Field(default="extraction")
    strict: Optional[bool] = Field(default=True)
    # Avoid shadowing BaseModel.schema by using an internal name with alias
    definition: Dict[str, Any] = Field(alias="schema")


class ExtractRequest(BaseModel):
    content_url: str
    model: str
    # Avoid shadowing BaseModel.schema at the top-level too; keep alias for payload compatibility
    response_schema: ExtractionSchema = Field(alias="schema")
    prompt: Optional[str] = Field(default="Analyze this content")
    filename: Optional[str] = Field(default="content.pdf")
    use_base64: Optional[bool] = Field(default=False)
    transforms: Optional[List[str]] = None


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest) -> Any:
    try:
        extractor = Extract(model=req.model)
        result = extractor.extract(
            content_url=req.content_url,
            filename=req.filename or "content.pdf",
            # Ensure payload matches OpenRouter's expected key names via aliases
            schema=req.response_schema.model_dump(by_alias=True),
            prompt=req.prompt or "Analyze this content",
            use_base64=bool(req.use_base64),
            transforms=req.transforms,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)