import os
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from src import CURRENT_MODEL, END_POINT_PROCESS_QUESTION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vector_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

DATA_FOLDER = "data"
DATA_FILE = "question.jsonl"
SUPPORTED_EXTENSIONS = [".jsonl"]  # Only process question.jsonl
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "conversations"
TOPIC = "Topic"
KEYWORDS = "Keywords"
MENTIONED_NAMES = "Mentioned_names"
CATEGORIES = "Categories"
CONTENT = "Content"

class ConversationMetadata(BaseModel):
    keywords: List[str] = Field(description="Lista słów kluczowych")
    mentioned_names: List[str] = Field(description="Lista imion rozmówców")
    main_topic: str = Field(description="Główny temat rozmowy")
    categories: List[str] = Field(description="Kategorie rozmowy")

class ProcessingResult(BaseModel):
    filename: str
    conversation_content: Dict[str, Any]
    metadata: ConversationMetadata
    processing_id: str

def generate_fallback_metadata(conversation_content: str) -> ConversationMetadata:
    """Generate minimal metadata when AI generation fails"""
    # Extract basic information from the conversation content
    try:
        # Try to parse the content and extract some basic details
        data = json.loads(conversation_content)

        # Extract keywords from text_full if available
        keywords = []
        if isinstance(data, dict) and 'text_full' in data:
            # Simple keyword extraction by splitting and taking unique words
            keywords = list(set(data['text_full'].lower().split()))[:5]

        # Extract mentioned names if available
        mentioned_names = []
        if isinstance(data, dict) and 'speaker_id' in data:
            mentioned_names = [data['speaker_id']]

        # Determine main topic and categories
        main_topic = "Rozmowa"
        categories = ["rozmowa"]

        if isinstance(data, dict) and 'type' in data:
            main_topic = data.get('type', 'Rozmowa')
            categories = [main_topic]

        return ConversationMetadata(
            keywords=keywords,
            mentioned_names=mentioned_names,
            main_topic=main_topic,
            categories=categories
        )
    except Exception as e:
        logger.warning(f"Fallback metadata generation failed: {e}")
        # Completely generic fallback
        return ConversationMetadata(
            keywords=["rozmowa"],
            mentioned_names=[],
            main_topic="Nieznany temat",
            categories=["rozmowa"]
        )

def generate_metadata(conversation_content: str) -> ConversationMetadata:
    """Generate metadata for conversation"""
    # Directly use fallback metadata generation
    return generate_fallback_metadata(conversation_content)

def process_file(file_path: str) -> List[ProcessingResult]:
    """Process single conversation file and return list of results"""
    logger.info(f"Processing file: {file_path}")

    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse each line as a separate JSON object for JSONL files
                    conversation_content = json.loads(line)

                    # Generate metadata for this conversation
                    metadata = generate_metadata(json.dumps(conversation_content))

                    # Create processing result for this conversation
                    result = ProcessingResult(
                        filename=Path(file_path).name,
                        conversation_content=conversation_content,
                        metadata=metadata,
                        processing_id=str(uuid.uuid4())
                    )

                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num} in {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num} in {file_path}: {e}")

        logger.info(f"Successfully processed {len(results)} conversations from {file_path}")
        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []

def batch_process(folder: str = DATA_FOLDER) -> List[ProcessingResult]:
    """Process all supported files in folder"""
    results = []
    folder_path = Path(folder)

    if not folder_path.exists():
        logger.error(f"Folder {folder} does not exist.")
        return results

    files_to_process = []
    for extension in SUPPORTED_EXTENSIONS:
        files_to_process.extend(folder_path.glob(f"*{extension}"))

    if not files_to_process:
        logger.warning(f"No supported files found in folder {folder}")
        return results

    logger.info(f"Found {len(files_to_process)} files to process")

    for file in files_to_process:
        result = process_file(str(file))
        if result:
            results.append(result)

    logger.info(f"Successfully processed {len(results)} of {len(files_to_process)} files")
    return results

def initialize_vector_db():
    """Initialize ChromaDB client and collection"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction()
        )
        logger.info(f"Vector database initialized at: {CHROMA_DB_PATH}")
        return client, collection
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        return None, None

def add_to_vector_db(collection, results: List[ProcessingResult]):
    """Add processed conversations to vector database"""
    if not collection:
        logger.error("No vector database collection available")
        return

    documents = []
    metadatas = []
    ids = []

    for result in results:
        doc_text = json.dumps(result.conversation_content, ensure_ascii=False)

        documents.append(doc_text)

        metadata = {
            TOPIC: result.metadata.main_topic,
            KEYWORDS: ", ".join(result.metadata.keywords[:20]),
            CATEGORIES: ", ".join(result.metadata.categories[:5]),
            MENTIONED_NAMES: ", ".join(result.metadata.mentioned_names)
        }
        metadatas.append(metadata)
        ids.append(result.processing_id)

    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(documents)} conversations to vector database")
    except Exception as e:
        logger.error(f"Error adding to vector database: {e}")

def search_vector_db(collection, query: str, n_results: int = 3):
    """Search vector database for similar conversations"""
    if not collection:
        logger.error("No vector database collection available")
        return None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        logger.info(f"Vector search performed for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        return None

def main():
    """Main program function"""
    logger.info("=== Conversation File Processor ===")

    logger.info("Initializing vector database...")
    _, collection = initialize_vector_db()

    results = batch_process()

    if results:
        if collection:
            logger.info("Adding conversations to vector database...")
            add_to_vector_db(collection, results)
            logger.info(f"Total conversations added: {len(results)}")
    else:
        logger.warning("No files were successfully processed.")

# Automatically process files when imported
if __name__ == "__main__" or __name__ == "src.vectordb":
    main()
