import typer
from typing import Optional, List
from phi.agent import Agent
from phi.model.groq import Groq
from phi.assistant import Assistant
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2, PgVector
from phi.embedder.huggingface import HuggingfaceCustomEmbedder
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

import os
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformerEmbedder(dimensions=384)

# Set up the database URL
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Create a knowledge base from pdf
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # vector_db=PgVector2(collection="recipes", db_url=db_url, embedder=HuggingfaceCustomEmbedder(model='sentence-transformers/all-MiniLM-L6-v2'))
    vector_db=PgVector2(
        collection="recipes", 
        db_url=db_url, 
        embedder=embedder,
    )
)

# Load the knowledge base
knowledge_base.load()

# Create a storage
storage = PgAgentStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    assistant = Agent(
        model=Groq(id="llama-3.3-70b-versatile",embedder=embedder),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(pdf_assistant)