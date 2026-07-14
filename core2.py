from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):           # schema
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object = Movie)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract the movie information from the given paragraph
    {format_instructions}"""
    ),
    ('human',"{paragraph}")
])

para = input("Enter the paragraph to analyze : ")
final_prompt = prompt.invoke(
    {"paragraph": para, 
     "format_instructions": parser.get_format_instructions()
    }
)


model = ChatMistralAI(
    model_name="mistral-small-2506"
)

response = model.invoke(final_prompt)

print(response.content)