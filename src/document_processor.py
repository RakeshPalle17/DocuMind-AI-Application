from __future__ import annotations

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def process_file(self, file_path: str, original_name: str) -> list[Document]:
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in (".txt", ".md"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            documents = loader.load()
        except Exception as exc:
            raise RuntimeError(f"Failed to load '{original_name}': {exc}") from exc

        for doc in documents:
            doc.metadata["source"] = original_name

        return self.splitter.split_documents(documents)
