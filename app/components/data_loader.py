import os
from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import load_vector_store, save_vector_store
from app.config.config import DB_FAISS_PATH

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def process_and_store_data():
    print("==================> Processing and storing data")
    try:
        logger.info("Making the vector store")
        documents = load_pdf_files()
        logger.info(f"Loaded {len(documents)} documents")
        text_chunks = create_text_chunks(documents)
        logger.info(f"Created {len(text_chunks)} text chunks")
        vector_store = save_vector_store(text_chunks)
        if vector_store:
            logger.info("Vector store created and saved successfully.")
        else:
            logger.error("Failed to create and save vector store.")

    except Exception as e:
        error_message = CustomException("Error occurred during data processing and storage.", e)
        logger.error(str(error_message))


if __name__ == "__main__":
    process_and_store_data()