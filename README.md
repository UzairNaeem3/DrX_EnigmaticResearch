# 🧪 Document Processing, RAG, Translation, and Summarization System

This repository contains a comprehensive system for processing documents into semantic chunks, building a vector database, implementing a Retrieval-Augmented Generation (RAG) system for efficient document question answering, translating documents between multiple languages, and generating high-quality document summaries.RetryClaude can make mistakes. Please double-check responses.

## 📚 Overview

The system consists of five main components:
1. An **Enhanced Document Chunker** that processes various document formats and breaks them into semantic chunks
2. A **Vector Database Builder** that creates embeddings for these chunks and stores them in a ChromaDB vector database
3. A **RAG System** that uses the vector database for retrieval and a large language model for answer generation
4. A **Translation Service** that translates documents between multiple languages while preserving formatting
5. A **Summarization System** that generates high-quality abstractive and extractive summaries of documents

## Features

- **Multi-format document processing**: Supports PDF, DOCX, XLSX, and CSV files
- **Semantic chunking**: Creates meaningful text chunks that respect document structure and semantic boundaries
- **High-quality embeddings**: Uses Nomic AI's embedding model for state-of-the-art text representations
- **Persistent vector database**: Stores embeddings in ChromaDB for efficient similarity search
- **Advanced RAG implementation**: Integrates retrieval with language model generation
- **Model caching**: Optimizes memory usage by caching models across instances
- **Performance monitoring**: Measures token processing speed across different tasks
- **Conversation history**: Maintains context for follow-up questions
- **Device optimization**: Automatically selects the appropriate device (CPU/GPU)
- **Fallback models**: Gracefully handles resource constraints with smaller models
- **Multi-language translation**: Supports translation between 100+ languages using efficient neural machine translation models
- **Format preservation**: Maintains document structure, tables, lists, and special formatting during translation
- **File translation**: Handles DOCX, PDF, and text files with page-by-page translation
- **Multiple summarization approaches**: Supports abstractive, bullet point, and multi-summary generation
- **Summary evaluation**: Evaluates summaries using ROUGE metrics and LLM-based comparison
- **Two-step summarization**: Generates and selects the best summary through a two-phase process
- **Confidence scoring**: Provides confidence metrics for generated summaries

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-processing-system.git
cd document-processing-system

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- tiktoken
- unstructured (for document parsing)
- sentence-transformers
- chromadb
- tqdm
- uuid
- json
- os
- torch
- transformers
- time
- functools
- fasttext (for language detection)
- PyPDF2 
- python-docx
- transformers
- rouge (for summary evaluation)
- nltk (for NLP tasks)
  
## Usage

### Processing Documents

```python
from document_chunker import EnhancedDocumentChunker

# Initialize the chunker
chunker = EnhancedDocumentChunker(
    chunk_size=512,
    chunk_overlap=50,
    tokenizer_name="cl100k_base",
    skip_if_empty=["Image", "Figure"],
)

# Process all documents in a directory
documents = chunker.process_directory("/path/to/your/documents")

# Create semantic chunks
chunks = chunker.process_and_chunk_documents(documents)

# Save chunks to disk
chunker.save_chunks(chunks)
```

### Building Vector Database

```python
from vector_db_builder import VectorDatabaseBuilder

# Initialize the vector database builder
db_builder = VectorDatabaseBuilder(
    chunks_dir="chunked_data",
    db_dir="vector_db",
    collection_name="MY_DOCUMENT_COLLECTION",
    embedding_model="nomic-ai/nomic-embed-text-v1.5"
)

# Build the vector database
db_builder.build_vector_database()
```

### Using the RAG System

```python
from rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem(
    db_dir="vector_db",
    collection_name="DR_X_Publications",
    llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    embedding_model="nomic-ai/nomic-embed-text-v1.5",
    retrieve_k=5,
    device='auto'
)

# Ask a question
answer = rag.answer_question("What are the key features of the document processing system?")
print(answer)

# Ask follow-up questions (uses conversation history)
follow_up_answer = rag.answer_question("Can you tell me more about the chunking process?")
print(follow_up_answer)

# Reset conversation history
rag.reset_conversation()
```

### Using the Translation Service

```python
from translation_service import TranslationService

# Initialize the translation service
translator = TranslationService(
    model_name="facebook/nllb-200-distilled-600M",
    device="auto",
    preserve_formatting=True
)

# Translate text
translated_text = translator.translate(
    text="Hello, how are you?", 
    target_language="ar"
)
print(translated_text)

# Translate a file
translated_file_path = translator.translate_file(
    file_path="document.docx",
    target_language="fr"
)
```

### Using the Summarization System

```python
from summarization_system import SummarizationSystem

# Initialize the summarization system
summarizer = SummarizationSystem(
    summaries_dir="summaries",
    llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    embedding_model="nomic-ai/nomic-embed-text-v1.5",
    device='auto'
)

# Load document chunks
chunks = summarizer.load_chunks("all_chunks.json")

# Group chunks by source
grouped_chunks = summarizer.group_chunks_by_source(chunks)

# Generate a summary for a specific document
document_chunks = grouped_chunks["example_document.pdf"]
summary_result = summarizer.summarize_document(document_chunks, method="abstractive")
print(summary_result["summary"])

# Generate multiple summaries and evaluate them
text = "Your text to summarize here..."
multiple_summaries = summarizer.generate_multiple_summaries(text, count=3)
evaluation = summarizer.evaluate_summaries_with_llm(text, multiple_summaries)
print(f"Best summary: {multiple_summaries[evaluation['best_summary_index']]}")

# Two-step summarization
two_step_result = summarizer.two_step_summarization(text)
print(f"Selected summary: {two_step_result['selected_summary']}")
```

## ⚙️ Methodology

### Document Processing

1. **Document Parsing**: The system uses the `unstructured` library to parse various document formats (PDF, DOCX, XLSX, CSV) and extract structured elements like text blocks, tables, and figures.

2. **Element Processing**: Each extracted element is processed and normalized:
   - Text is cleaned to remove extra whitespace and normalize quotes
   - Empty elements of specified types are filtered out
   - Element metadata is preserved

3. **Semantic Chunking**: Documents are divided into semantically meaningful chunks:
   - Chunks respect semantic boundaries like headings and section breaks
   - Chunks maintain a specified token size (default: 512 tokens)
   - Optional overlap between chunks ensures context continuity (default: 50 tokens)
   - Special handling for tables and other complex elements

4. **Output**: Processed chunks are saved as JSON files with metadata including:
   - Source document
   - Page numbers
   - Chunk number
   - Token count
   - Element types
   - HTML representation of tables (when applicable)

### Vector Database Creation

1. **Embedding Generation**: The system uses the Nomic AI embedding model (`nomic-ai/nomic-embed-text-v1.5`) to generate high-quality vector representations of text chunks:
   - State-of-the-art embeddings that capture semantic meaning
   - Embeddings are generated in batches to optimize memory usage

2. **Database Storage**: Embeddings are stored in a ChromaDB vector database:
   - Uses cosine similarity for semantic matching
   - Maintains document metadata for filtering and retrieval
   - Persists to disk for future use

### RAG System Implementation

1. **Query Processing**: The system processes user queries by:
   - Generating embeddings for the query
   - Enhancing follow-up questions with conversation history
   - Measuring token processing speed

2. **Retrieval**: Relevant document chunks are retrieved by:
   - Performing similarity search in the vector database
   - Retrieving top-k chunks (default: 5)
   - Including metadata and distance scores

3. **Context Formatting**: Retrieved chunks are formatted into a prompt context:
   - Including document sources and page numbers
   - Handling special elements like tables
   - Adding conversation history for contextual awareness

4. **Answer Generation**: The system generates answers using:
   - Model-specific prompt formatting (Llama-specific vs generic)
   - Dynamic token allocation based on context window
   - Temperature control for deterministic answers
   - Measurement of token processing speed

5. **Performance Optimization**:
   - Model caching to avoid repeated loading
   - Device selection (CUDA/CPU) based on availability
   - Fallback to smaller models when resources are constrained
   - 8-bit quantization for CUDA implementations

### Translation Service

1. **Model Loading**: The system loads neural machine translation models:
   - Primary model: `facebook/nllb-200-distilled-600M` (supports 200+ languages)
   - Fallback model: `facebook/m2m100_418M` (smaller alternative)
   - Intelligent model caching to avoid reloading

2. **Language Processing**:
   - Automatic language detection using FastText
   - Mapping between common language names and model-specific language codes
   - Support for 200+ language pairs

3. **Chunking and Processing**:
   - Smart text chunking to handle long documents within model context limits
   - Sentence and paragraph boundary preservation
   - Performance monitoring with token processing metrics

4. **Format Preservation**:
   - Structure retention for paragraphs, lists, and bullet points
   - Table format preservation during translation
   - Code block handling to prevent translation of code snippets
   - Styling and formatting retention in Word documents

5. **File Processing**:
   - Word document (.docx) translation with formatting preservation
   - PDF translation with page-by-page extraction and processing
   - Output organization with individual page files and complete translations

6. **📁 Translated Files**:
   - Some of the translated files can be found in the [`translated_files/`](https://github.com/UzairNaeem3/DrX_EnigmaticResearch/tree/master/translated_files) directory.
   
### Summarization System Implementation

1. **Summary Generation**: The system generates summaries through multiple approaches:
   - **Abstractive Summarization**: Uses the LLM to generate coherent, fluent summaries that restate the source content
   - **Bullet Point Summarization**: Creates concise, structured bullet points highlighting key concepts
   - **Multi-Summary Generation**: Produces multiple alternative summaries to capture different perspectives or styles

2. **Document Processing**:
   - Processes document chunks by source and metadata
   - Combines related chunks for comprehensive document summarization
   - Maintains structural elements like tables and lists

3. **Evaluation and Selection**:
   - **ROUGE Evaluation**: Calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics to compare summaries against references
   - **LLM-based Evaluation**: Uses the language model to analyze and compare multiple summaries
   - **Confidence Scoring**: Assigns confidence metrics to generated summaries

4. **Two-Step Summarization Process**:
   - First phase: Generates multiple candidate summaries with different approaches
   - Second phase: Evaluates candidates and selects the optimal summary
   - Provides explanation for selection criteria and decision process

5. **Performance Optimization**:
   - Token processing measurement across summarization tasks
   - Optimized context window management
   - Caches models to improve performance across multiple summarization tasks
  
## Models

### Large Language Model (LLM)

The system primarily uses the **Meta-Llama-3-8B-Instruct** model for answer generation:

- **Provider**: Meta AI
- **Model**: Meta-Llama-3-8B-Instruct
- **Size**: 8 billion parameters
- **Context Window**: 8,192 tokens
- **Implementation**: Used via HuggingFace's Transformers library

For systems with limited resources, the following fallback models are supported:
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B parameters)
- **google/flan-t5-base** (smaller model)

### Embedding Model

The system uses the **nomic-ai/nomic-embed-text-v1.5** model for generating text embeddings:

- **Provider**: Nomic AI
- **Model**: nomic-embed-text-v1.5
- **Embedding Size**: 768 dimensions
- **Context Window**: Supports up to 8192 tokens
- **Performance**: State-of-the-art performance on various text embedding benchmarks
- **Implementation**: Used via the sentence-transformers library

### Translation Model

The system uses the **facebook/nllb-200-distilled-600M** model for translations:

- **Provider**: Meta AI/Facebook
- **Model**: NLLB (No Language Left Behind) 200-language distilled model
- **Size**: 600M parameters (distilled from larger model)
- **Languages**: Supports 200+ languages and language pairs
- **Performance**: State-of-the-art quality for a model of its size
- **Implementation**: Used via HuggingFace's Transformers library

For systems with limited resources, the following fallback model is supported:
- **facebook/m2m100_418M** (smaller model for 100 languages)

### Summarization Models

The summarization system utilizes the same Large Language Models as the RAG system for generating and evaluating summaries:

- **Primary LLM**: Meta-Llama-3-8B-Instruct
- **Fallback Models**: TinyLlama-1.1B-Chat-v1.0 or other smaller models when resources are limited

For summary evaluation, the system also incorporates:

- **ROUGE Metrics**: Implements ROUGE-1, ROUGE-2, and ROUGE-L for statistical evaluation
- **Embedding Comparison**: Uses sentence-transformers embeddings to measure semantic similarity

## Performance Monitoring

The system includes a `measure_token_processing` decorator that monitors the performance of token-intensive operations:

- Tracks input and output tokens
- Measures processing time
- Calculates tokens-per-second metrics
- Reports performance statistics to the console

## Customization

### Chunking Parameters

- `chunk_size`: Maximum number of tokens per chunk (default: 512)
- `chunk_overlap`: Number of tokens to overlap between chunks (default: 50)
- `tokenizer_name`: Tokenizer for counting tokens (default: "cl100k_base")
- `skip_if_empty`: Element types to skip if they contain no text (default: ["Image", "Figure"])

### Vector Database Parameters

- `chunks_dir`: Directory where chunks are stored (default: "chunked_data")
- `db_dir`: Directory where the vector database is stored (default: "vector_db")
- `collection_name`: Name of the ChromaDB collection (default: "DR_X_Publications")
- `embedding_model`: Model used for generating embeddings (default: "nomic-ai/nomic-embed-text-v1.5")

### RAG System Parameters

- `db_dir`: Directory of the vector database (default: "vector_db")
- `collection_name`: Name of the ChromaDB collection (default: "DR_X_Publications")
- `llm_model_name`: Name of the LLM model (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `embedding_model`: Name of the embedding model (default: "nomic-ai/nomic-embed-text-v1.5")
- `retrieve_k`: Number of chunks to retrieve (default: 5)
- `device`: Computing device to use ("auto", "cuda", or "cpu")

### Translation Service Parameters

- `model_name`: Translation model to use (default: "facebook/nllb-200-distilled-600M")
- `device`: Computing device to use ("auto", "cuda", or "cpu")
- `preserve_formatting`: Whether to maintain document formatting (default: True)

### Summarization System Parameters

- `summaries_dir`: Directory where summaries are stored (default: "summaries")
- `llm_model_name`: LLM model for generating summaries (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `embedding_model`: Embedding model for semantic comparisons (default: "nomic-ai/nomic-embed-text-v1.5")
- `device`: Computing device to use ("auto", "cuda", or "cpu")
- `max_length`: Maximum length for generated summaries (default: 500 tokens)
