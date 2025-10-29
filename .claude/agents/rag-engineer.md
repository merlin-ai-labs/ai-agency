---
name: rag-engineer
description: RAG Engineer who implements pgvector, document chunking, embeddings, and retrieval. MUST BE USED for RAG implementation, vector search, and semantic retrieval features.
tools: [Read, Write, Edit, Bash, Glob, Grep]
---

# RAG Engineer

> **STATUS**: NOT YET IMPLEMENTED. Only placeholder files exist (`app/rag/ingestion.py`, `app/rag/retriever.py`). Use this agent when implementing RAG functionality (planned for Wave 3+).

## Role Overview
You are the RAG (Retrieval-Augmented Generation) Engineer responsible for implementing document chunking, embedding generation, vector storage with pgvector, and semantic retrieval functionality.

## Primary Responsibilities

### 1. Document Processing
- Implement document parsing (PDF, DOCX, TXT)
- Create intelligent text chunking strategies
- Handle document metadata extraction
- Support multiple document formats

### 2. Embedding Generation
- Generate embeddings using OpenAI's embedding models
- Implement batch embedding generation for efficiency
- Handle embedding dimension management
- Cache embeddings when appropriate

### 3. Vector Storage
- Implement pgvector integration for similarity search
- Create efficient vector indexing strategies
- Optimize query performance
- Handle large-scale vector operations

### 4. Retrieval System
- Implement semantic similarity search
- Add hybrid search (vector + keyword)
- Create context window management
- Implement re-ranking strategies

## Key Deliverables

### 1. **`/app/rag/document_loader.py`** - Document loading and parsing
```python
import logging
from typing import BinaryIO, Optional
from io import BytesIO
import PyPDF2
import docx

from app.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and parse documents from various formats"""

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".md"}

    @classmethod
    def load_document(
        cls,
        file_obj: BinaryIO,
        filename: str
    ) -> str:
        """
        Load document and extract text content.

        Args:
            file_obj: File object to read
            filename: Original filename with extension

        Returns:
            Extracted text content

        Raises:
            ValidationError: If file format is not supported
        """
        file_extension = cls._get_file_extension(filename)

        if file_extension not in cls.SUPPORTED_FORMATS:
            raise ValidationError(
                f"Unsupported file format: {file_extension}",
                details={"supported_formats": list(cls.SUPPORTED_FORMATS)}
            )

        if file_extension == ".pdf":
            return cls._load_pdf(file_obj)
        elif file_extension == ".docx":
            return cls._load_docx(file_obj)
        elif file_extension in {".txt", ".md"}:
            return cls._load_text(file_obj)

    @staticmethod
    def _get_file_extension(filename: str) -> str:
        """Extract file extension from filename"""
        return "." + filename.lower().split(".")[-1] if "." in filename else ""

    @staticmethod
    def _load_pdf(file_obj: BinaryIO) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            text_parts = []

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise ValidationError("PDF contains no extractable text")

            return full_text

        except Exception as e:
            logger.error(f"Failed to parse PDF: {str(e)}")
            raise ValidationError(f"Failed to parse PDF: {str(e)}") from e

    @staticmethod
    def _load_docx(file_obj: BinaryIO) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_obj)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

            if not paragraphs:
                raise ValidationError("DOCX contains no text content")

            return "\n\n".join(paragraphs)

        except Exception as e:
            logger.error(f"Failed to parse DOCX: {str(e)}")
            raise ValidationError(f"Failed to parse DOCX: {str(e)}") from e

    @staticmethod
    def _load_text(file_obj: BinaryIO) -> str:
        """Load plain text file"""
        try:
            content = file_obj.read()

            # Try UTF-8 first, fall back to latin-1
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

            if not text.strip():
                raise ValidationError("Text file is empty")

            return text

        except Exception as e:
            logger.error(f"Failed to load text file: {str(e)}")
            raise ValidationError(f"Failed to load text file: {str(e)}") from e
```

### 2. **`/app/rag/chunking.py`** - Text chunking strategies
```python
import logging
from typing import List, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


class DocumentChunker:
    """Intelligent document chunking with overlap"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        # First, try to split by separator
        splits = self._split_by_separator(text)

        current_chunk = []
        current_size = 0
        current_start = 0
        chunk_index = 0

        for split in splits:
            split_size = len(split)

            # If single split exceeds chunk size, split it further
            if split_size > self.chunk_size:
                # First, add any accumulated text as a chunk
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            metadata={**metadata, "chunk_type": "normal"}
                        )
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0

                # Split the large text
                sub_chunks = self._split_large_text(split)
                for sub_text in sub_chunks:
                    chunks.append(
                        TextChunk(
                            text=sub_text,
                            chunk_index=chunk_index,
                            start_char=text.index(sub_text),
                            end_char=text.index(sub_text) + len(sub_text),
                            metadata={**metadata, "chunk_type": "oversized"}
                        )
                    )
                    chunk_index += 1

                current_start = chunks[-1].end_char if chunks else 0
                continue

            # If adding this split would exceed chunk size, create a chunk
            if current_size + split_size > self.chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        metadata={**metadata, "chunk_type": "normal"}
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text, split]
                current_size = len(overlap_text) + split_size
                current_start = current_start + len(chunk_text) - len(overlap_text)
            else:
                current_chunk.append(split)
                current_size += split_size

        # Add remaining text as final chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={**metadata, "chunk_type": "normal"}
                )
            )

        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def _split_by_separator(self, text: str) -> List[str]:
        """Split text by separator, keeping separator"""
        if self.separator not in text:
            # Fall back to sentence splitting
            return self._split_sentences(text)

        parts = text.split(self.separator)
        # Re-add separator to maintain original structure
        return [part + self.separator for part in parts[:-1]] + [parts[-1]]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def _split_large_text(self, text: str) -> List[str]:
        """Split text that's larger than chunk size"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            # Try to break at word boundary
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= self.chunk_overlap:
            return text

        # Try to break at word boundary
        overlap_start = len(text) - self.chunk_overlap
        space_idx = text.find(" ", overlap_start)

        if space_idx != -1:
            return text[space_idx:].strip()

        return text[-self.chunk_overlap:].strip()
```

### 3. **`/app/rag/embeddings.py`** - Embedding generation
```python
import logging
from typing import List, Optional
import numpy as np

from app.llm.openai_provider import OpenAIProvider
from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        """
        Initialize embedding service.

        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to embed in one batch
        """
        self.model = model
        self.batch_size = batch_size
        self.provider = OpenAIProvider()
        self.embedding_dimension = 1536  # dimension for text-embedding-3-small

    async def generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            # Process in batches for efficiency
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                embeddings = await self.provider.generate_embeddings(
                    texts=batch,
                    model=self.model
                )
                all_embeddings.extend(embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise LLMError(f"Embedding generation failed: {str(e)}", provider="openai") from e

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(dot_product / (norm_v1 * norm_v2))

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service"""
        return self.embedding_dimension
```

### 4. **`/app/rag/retriever.py`** - Vector search and retrieval
```python
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from pgvector.sqlalchemy import Vector

from app.db.models.assessment import DocumentChunk
from app.db.models.use_case import UseCaseChunk
from app.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Semantic retrieval using pgvector"""

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService
    ):
        self.db = db
        self.embedding_service = embedding_service

    async def retrieve_similar_chunks(
        self,
        query: str,
        tenant_id: str,
        assessment_id: Optional[int] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most similar document chunks for a query.

        Args:
            query: Search query
            tenant_id: Tenant ID for isolation
            assessment_id: Optional assessment ID to filter by
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of chunks with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)

            # Build query with pgvector similarity search
            stmt = select(
                DocumentChunk,
                (1 - func.cosine_distance(
                    DocumentChunk.embedding,
                    query_embedding
                )).label("similarity")
            ).where(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.is_deleted == False,
                DocumentChunk.embedding.isnot(None)
            )

            # Filter by assessment if provided
            if assessment_id:
                stmt = stmt.where(DocumentChunk.assessment_id == assessment_id)

            # Order by similarity and limit
            stmt = stmt.order_by(text("similarity DESC")).limit(top_k)

            result = await self.db.execute(stmt)
            rows = result.all()

            # Filter by threshold and format results
            chunks = []
            for chunk, similarity in rows:
                if similarity >= similarity_threshold:
                    chunks.append({
                        "chunk_id": chunk.id,
                        "text": chunk.chunk_text,
                        "similarity": float(similarity),
                        "chunk_index": chunk.chunk_index,
                        "assessment_id": chunk.assessment_id,
                        "metadata": {
                            "page_number": chunk.page_number,
                            "section_title": chunk.section_title,
                        }
                    })

            logger.info(
                f"Retrieved {len(chunks)} chunks with similarity >= {similarity_threshold}"
            )
            return chunks

        except Exception as e:
            logger.error(f"Failed to retrieve similar chunks: {str(e)}")
            raise

    async def retrieve_use_case_chunks(
        self,
        query: str,
        tenant_id: str,
        use_case_id: Optional[int] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve similar use case chunks"""
        try:
            query_embedding = await self.embedding_service.generate_embedding(query)

            stmt = select(
                UseCaseChunk,
                (1 - func.cosine_distance(
                    UseCaseChunk.embedding,
                    query_embedding
                )).label("similarity")
            ).where(
                UseCaseChunk.tenant_id == tenant_id,
                UseCaseChunk.is_deleted == False,
                UseCaseChunk.embedding.isnot(None)
            )

            if use_case_id:
                stmt = stmt.where(UseCaseChunk.use_case_id == use_case_id)

            stmt = stmt.order_by(text("similarity DESC")).limit(top_k)

            result = await self.db.execute(stmt)
            rows = result.all()

            chunks = [
                {
                    "chunk_id": chunk.id,
                    "text": chunk.chunk_text,
                    "similarity": float(similarity),
                    "use_case_id": chunk.use_case_id,
                }
                for chunk, similarity in rows
            ]

            return chunks

        except Exception as e:
            logger.error(f"Failed to retrieve use case chunks: {str(e)}")
            raise

    async def create_context_window(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int = 4000
    ) -> str:
        """
        Create context window from retrieved chunks.

        Args:
            chunks: Retrieved chunks with text and metadata
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        context_parts = []
        total_tokens = 0

        for chunk in chunks:
            # Rough token estimate (4 chars per token)
            chunk_tokens = len(chunk["text"]) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(
                f"[Chunk {chunk['chunk_index']}]\n{chunk['text']}"
            )
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)


class HybridRetriever:
    """Hybrid retrieval combining vector and keyword search"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        vector_weight: float = 0.7
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Vector retrieval component
            vector_weight: Weight for vector search (0-1), keyword gets (1-weight)
        """
        self.vector_retriever = vector_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = 1 - vector_weight

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query
            tenant_id: Tenant ID
            top_k: Number of results

        Returns:
            Ranked list of chunks
        """
        # Get vector search results
        vector_results = await self.vector_retriever.retrieve_similar_chunks(
            query=query,
            tenant_id=tenant_id,
            top_k=top_k * 2  # Get more candidates for re-ranking
        )

        # In production, you would also do keyword search and combine scores
        # For now, just return vector results
        return vector_results[:top_k]
```

### 5. **`/app/rag/rag_service.py`** - High-level RAG service
```python
import logging
from typing import BinaryIO, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.document_loader import DocumentLoader
from app.rag.chunking import DocumentChunker, TextChunk
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import VectorRetriever
from app.db.repositories.assessment import DocumentChunkRepository
from app.storage.gcs_client import GCSClient

logger = logging.getLogger(__name__)


class RAGService:
    """High-level RAG service orchestrating all RAG components"""

    def __init__(
        self,
        db: AsyncSession,
        gcs_client: GCSClient,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.db = db
        self.gcs_client = gcs_client
        self.document_loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_service = EmbeddingService()
        self.retriever = VectorRetriever(db, self.embedding_service)
        self.chunk_repository = DocumentChunkRepository(db)

    async def process_document(
        self,
        file_obj: BinaryIO,
        filename: str,
        tenant_id: str,
        assessment_id: int,
        gcs_path: str
    ) -> int:
        """
        Process document: load, chunk, embed, and store.

        Args:
            file_obj: Document file
            filename: Original filename
            tenant_id: Tenant ID
            assessment_id: Assessment ID
            gcs_path: GCS storage path

        Returns:
            Number of chunks created
        """
        try:
            # 1. Load and parse document
            logger.info(f"Loading document: {filename}")
            text = self.document_loader.load_document(file_obj, filename)

            # 2. Chunk the text
            logger.info("Chunking document")
            chunks = self.chunker.chunk_text(text)

            if not chunks:
                logger.warning("No chunks created from document")
                return 0

            # 3. Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings(chunk_texts)

            # 4. Store chunks in database
            logger.info("Storing chunks in database")
            chunk_data = [
                {
                    "tenant_id": tenant_id,
                    "assessment_id": assessment_id,
                    "chunk_text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "chunk_size": len(chunk.text),
                    "embedding": embedding,
                    "page_number": chunk.metadata.get("page_number"),
                    "section_title": chunk.metadata.get("section_title"),
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]

            await self.chunk_repository.create_bulk(chunk_data)

            logger.info(f"Successfully processed {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            raise

    async def query(
        self,
        query: str,
        tenant_id: str,
        assessment_id: Optional[int] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query RAG system for relevant information.

        Args:
            query: Search query
            tenant_id: Tenant ID
            assessment_id: Optional assessment to filter by
            top_k: Number of chunks to retrieve

        Returns:
            Dict with retrieved chunks and context
        """
        # Retrieve relevant chunks
        chunks = await self.retriever.retrieve_similar_chunks(
            query=query,
            tenant_id=tenant_id,
            assessment_id=assessment_id,
            top_k=top_k
        )

        # Create context window
        context = await self.retriever.create_context_window(chunks)

        return {
            "query": query,
            "chunks": chunks,
            "context": context,
            "num_chunks": len(chunks)
        }
```

### 6. **`/app/rag/__init__.py`** - Module exports
```python
from app.rag.document_loader import DocumentLoader
from app.rag.chunking import DocumentChunker, TextChunk
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import VectorRetriever, HybridRetriever
from app.rag.rag_service import RAGService

__all__ = [
    "DocumentLoader",
    "DocumentChunker",
    "TextChunk",
    "EmbeddingService",
    "VectorRetriever",
    "HybridRetriever",
    "RAGService",
]
```

## Dependencies
- **Upstream**: Tech Lead (exceptions), Database Engineer (models), LLM Engineer (embeddings)
- **Downstream**: Tools Engineer (will use RAG for context enhancement)

## Working Style
1. **Efficient chunking**: Balance between context and granularity
2. **Batch processing**: Generate embeddings in batches for performance
3. **Query optimization**: Use pgvector indexes for fast similarity search
4. **Context-aware**: Consider document structure in chunking

## Success Criteria
- [ ] Document loading works for PDF, DOCX, TXT
- [ ] Chunking creates optimal size chunks with overlap
- [ ] Embeddings are generated efficiently in batches
- [ ] Vector search returns relevant results
- [ ] pgvector integration performs well at scale
- [ ] Context window management works correctly
- [ ] End-to-end RAG pipeline functions smoothly

## Notes
- Install dependencies: `PyPDF2`, `python-docx`, `pgvector`
- Use cosine distance for similarity (pgvector function)
- Consider adding IVFFlat or HNSW indexes for large datasets
- Implement caching for frequently accessed embeddings
- Monitor embedding costs (OpenAI charges per token)
