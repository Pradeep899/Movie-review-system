import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Movie(BaseModel):           # schema
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract the movie information from the given paragraph
    {format_instructions}"""
     ),
    ('human', "{paragraph}")
])

model = ChatMistralAI(
    model_name="mistral-small-2506"
)

st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
    .stTextArea textarea { border-radius: 10px; }
    div.stButton > button { border-radius: 10px; width: 100%; }
    .movie-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 10px;
    }
    .movie-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .movie-subtitle {
        color: #9aa0a6;
        font-size: 15px;
        margin-bottom: 14px;
    }
    .badge {
        display: inline-block;
        background-color: rgba(147, 112, 219, 0.2);
        color: #c9a9f2;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 2px 6px 2px 0px;
        font-size: 13px;
    }
    .field-label {
        color: #9aa0a6;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Movie Info Extractor")
st.caption("Paste a paragraph about a movie and get structured details out of it.")

para = st.text_area("Enter the paragraph to analyze", height=200, placeholder="Paste your paragraph here...")

analyze = st.button("Extract")

if analyze:
    if not para.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting movie details..."):
            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions": parser.get_format_instructions()
            })
            response = model.invoke(final_prompt)

            try:
                movie = parser.parse(response.content)
            except Exception:
                movie = None

        st.markdown("---")

        if movie:
            genre_badges = "".join(f"<span class='badge'>{g}</span>" for g in movie.genre)
            cast_text = ", ".join(movie.cast) if movie.cast else "Not Mentioned"
            rating_text = f"⭐ {movie.rating}" if movie.rating is not None else "Not Mentioned"
            year_text = movie.release_year if movie.release_year else "Not Mentioned"
            director_text = movie.director if movie.director else "Not Mentioned"

            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{movie.title}</div>
                    <div class="movie-subtitle">{year_text} · Directed by {director_text}</div>
                    <div>{genre_badges}</div>
                    <div class="field-label">Rating</div>
                    <div>{rating_text}</div>
                    <div class="field-label">Cast</div>
                    <div>{cast_text}</div>
                    <div class="field-label">Summary</div>
                    <div>{movie.summary}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View raw JSON"):
                st.json(movie.model_dump())
        else:
            st.subheader("Extracted Information")
            st.write(response.content)