# 🧪 The Enigmatic Research of Dr. X

## 📚 Project Overview
This project explores the mysterious disappearance of Dr. X, a renowned researcher. Using their collection of publications in various formats, a comprehensive NLP pipeline was developed to extract, analyze, and interact with the information using a RAG (Retrieval-Augmented Generation) approach. The solution includes text extraction, chunking, embedding, translation, summarization, and a Q&A system powered by LLMs.

---

## ⚙️ Methodology

### 1. 🔍 Text Extraction
- Handled various document formats including `.pdf`, `.docx`, and `.doc`.
- Experimented with multiple parsing approaches to evaluate accuracy and layout preservation.
- Selected **Unstructured.io** due to its excellent performance in:
  - Extracting structured text from complex layouts
  - Preserving contextual order of elements
  - Handling tables and page breaks effectively
- Used the `hi_res` strategy in `partition_pdf` and `partition_docx` for high-fidelity extraction.
- Converted extracted elements to JSON format for easier downstream processing.
- Cleaned each element to remove extra whitespace and non-informative content.
- Filtered out empty elements of specific types (e.g., headers or footers with no meaningful text).

### 2. 🧩 Chunking

- After extracting text elements, each document is split into **semantic chunks** for better context preservation.
- Used a **custom chunker** that processes elements based on their type, content, and token count.
- Chunks were created with a maximum of `512 tokens`, allowing an **overlap of 50 tokens** between chunks for smoother transitions.
- Special handling was done for:
  - **Tables**, which were preserved as HTML and grouped into chunks without splitting them.
  - **Semantic boundaries** like headings and section titles, which trigger new chunks to maintain logical structure.
- Elements were added to a chunk until the token limit was reached; if exceeded, a new chunk started with an overlapping part of the previous one.
- Each chunk includes:
  - `text`: the combined content of elements
  - `pages`: where the chunk originated
  - `chunk_number`, `token_count`, and `element_types`
  - Optional: `tables` (in HTML)
- Cleaned each chunk’s text and skipped empty or non-informative types (like `Image`, `Figure`).
- Finally, chunks were saved in both **individual files per document** and a single **consolidated file** for easy access.

### 3. 🧠 Embedding and Vector Database

- Used the [**Nomic Embed Text v1.5**](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) model from HuggingFace via `SentenceTransformer` for generating high-quality embeddings.
  - Model was selected for its **open-source availability**, **speed**, and **strong performance** in semantic understanding.
- Loaded all semantic chunks and encoded them into vector representations using this embedding model.
- To manage these vectors and support fast similarity searches, I used **ChromaDB** as the vector database.

#### 📦 Why ChromaDB?
- **Simple and lightweight** local setup — no need for external cloud services.
- Supports **persistent storage** of vectors and metadata.
- Offers **cosine similarity** for efficient nearest-neighbor search.
- Integrates well with Python and is actively maintained.

#### 🧱 Methodology:
- Loaded document chunks from a JSON file (`all_chunks.json`).
- Batched the chunks (100 per batch) to avoid memory bottlenecks during embedding generation.
- For each batch:
  - Generated embeddings using the Nomic model.
  - Constructed rich metadata for each chunk, including:
    - `source`, `page_number`, `chunk_number`, `token_count`
    - Optional: `element_types` and embedded `tables` (as stringified HTML)
  - Stored embeddings, texts, and metadata into ChromaDB collection.
- After all chunks were processed, validated the database by checking the entry count.

> ✅ Result: A fully searchable, persistent vector database containing rich, semantically embedded document content ready for retrieval-augmented generation (RAG).

### 4. 💬 RAG-Based Q&A System
The **RAG-Based Q&A System** utilizes **Retrieval-Augmented Generation (RAG)** to enhance question answering by combining relevant information retrieved from a vector database with a language model for generating responses. The system consists of several components and workflows to provide accurate and relevant answers.

#### Key Features

- **Embedding Model**: Transforms input text into fixed-size vectors for semantic comparison.
- **Vector Database**: Stores embedded documents for efficient similarity search.
- **Large Language Model (LLM)**: Generates answers based on retrieved context from the database.
- **Conversation History**: Maintains context for follow-up questions to enhance conversation flow.

#### Class: `RAGSystem`

##### Initialization

```python
class RAGSystem:
    def __init__(self, db_dir, collection_name, llm_model_name, embedding_model, retrieve_k, device):
        # Initialize parameters like database path, collection name, model names, and device
        # Load ChromaDB client and initialize LLM, tokenizer, embedding model
```
- **ChromaDB** is used to store and query the vector database.
- The **LLM** is loaded based on the selected model (e.g., Meta Llama or fallback TinyLlama).
- The **embedding model** is used for converting text to embeddings, allowing semantic search.

##### Embedding and Search
- `generate_embedding(text)`: Converts text into vector embeddings using the embedding model.
- `retrieve_relevant_chunks(query)`: Retrieves the most relevant chunks from the vector database based on a query’s embedding.

##### Context Formatting
- `format_context(chunks)`: Formats the retrieved chunks into a structured context that is fed into the LLM for generating responses.
- `generate_prompt(query, context)`: Combines the query and context into a prompt tailored for the LLM.

##### Answer Generation
- `generate_answer(prompt)`: Uses the LLM to generate an answer based on the formatted prompt.
- `answer_question(query)`: Main method for answering queries by retrieving relevant chunks, formatting context, and generating an answer.

##### Conversation Flow
- The system maintains a conversation history, which can be used for follow-up questions to enhance the context of the query.
- `_enhance_follow_up_query(query)`: Adds context from recent conversations to improve follow-up questions.
- `reset_conversation()`: Clears the conversation history.

##### Example Usage
```python
rag = RAGSystem(
    db_dir="vector_db", 
    collection_name="DR_X_Publications",
    llm_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    embedding_model="nomic-ai/nomic-embed-text-v1.5",
    retrieve_k=5,
    device="auto"
)
```
