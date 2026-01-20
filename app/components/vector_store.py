from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading FAISS vector store from {DB_FAISS_PATH}")
            vector_store = FAISS.load_local(DB_FAISS_PATH,
                                            embedding_model,
                                            allow_dangerous_deserialization=True)
            logger.info("FAISS vector store loaded successfully.")
            return vector_store
        else:
            logger.warning(f"FAISS vector store path {DB_FAISS_PATH} does not exist.")
            return None

    except Exception as e:
        error_message = CustomException("Error occurred while loading vector store.", e)
        logger.error(str(error_message))
        return None


# create new vector store
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to create vector store.")

        logger.info("Genertating new vector store")

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving FAISS vector store to local disk")

        db.save_local(DB_FAISS_PATH)
        logger.info("FAISS vector store saved successfully.")
        return db

    except Exception as e:
        error_message = CustomException("Error occurred while saving vector store.", e)
        logger.error(str(error_message))
        return None