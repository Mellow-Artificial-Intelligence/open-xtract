from dotenv import load_dotenv
from open_xtract import OpenXtract
from pydantic import BaseModel, Field

load_dotenv()


def main() -> None:

    class Resume(BaseModel):
        name: str = Field(description="The name of the person")
        email: str = Field(description="The email of the person")
        phone: str = Field(description="The phone number of the person")
        address: str = Field(description="The address of the person")
        city: str = Field(description="The city of the person")
        state: str = Field(description="The state of the person")
        zip: str = Field(description="The zip code of the person")

    # Use xAI Grok
    ox = OpenXtract(model="xai:grok-4")
    #ox = OpenXtract(model="openai:gpt-5-nano")
    result = ox.extract(
        "This is a test, I live at 123 Main St, Anytown, USA 12345. My email is test@test.com and my phone number is 123-456-7890.",
        Resume,
    )
    print(result)


if __name__ == "__main__":
    main()
