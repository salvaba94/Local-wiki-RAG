from langchain_community.embeddings import HuggingFaceEmbeddings
from enum import Enum


class Embeddings(Enum):
    HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        model_kwargs={"device": "cuda"},
    )
    
