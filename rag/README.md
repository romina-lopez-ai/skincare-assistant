# RAG Module: Skincare Model

This module implements the Retrieval Augmented Generation (RAG) pipeline for the skincare model, enabling it to answer skin disease-related questions by retrieving context from external documents (PubMed abstracts and local PDFs) rather than relying solely on its pre-trained knowledge. This approach significantly reduces hallucination and grounds the model's responses in factual, verifiable information.

## Key Components

The RAG module is designed with a modular architecture, each component serving a specific role in the overall retrieval and generation process:

### `config.py`

Centralizes all configurable parameters for the RAG module, including paths for ChromaDB (local or cloud), embedding and re-ranker model names, Ollama LLM settings (temperature, max tokens), RAG hyperparameters (relevance thresholds, chunk sizes), query expansion suffixes, and PubMed indexing specifics (Entrez email, medical keywords, topics).

### `model_loader.py`

Manages the loading and provides singleton access to the core AI models:
- **Embedding Model**: Used to convert text into numerical vectors for semantic search.
- **Re-ranker Model (FlashRank)**: A cross-encoder that re-ranks retrieved documents to improve relevance before passing them to the LLM.
- **Ollama LLM Model**: The fine-tuned language model responsible for generating final answers.

### `vector_store.py`

Handles the interaction with the vector database (ChromaDB) and the document store. It utilizes LangChain's `ParentDocumentRetriever` to manage document chunks effectively, storing smaller "child" chunks for embedding and retrieval, and larger "parent" chunks for providing rich context to the LLM. It supports both local and cloud ChromaDB instances.

### `query_processor.py`

Implements the core logic for processing user queries and refining retrieved documents. This includes:
- **Query Expansion**: Appending clinical keywords to initial queries for broader retrieval.
- **Reciprocal Rank Fusion (RRF)**: Merging results from multiple retrieval methods (e.g., dense and BM25) to create a robust ranked list of candidates.
- **FlashRank Re-ranking**: Applying a cross-encoder model to the top RRF candidates for a more accurate relevance score.
- **Score Filtering**: Discarding documents below a predefined relevance threshold to ensure only high-quality context is used.

### `prompt_factory.py`

Constructs the final prompt that is sent to the Ollama LLM. This component ensures the prompt format is consistent with the model's training and gracefully handles cases where no relevant context is found, instructing the LLM to admit it lacks information.

## RAG Core Components (Replaced LangGraph Nodes)

The RAG pipeline now leverages two main components for retrieval and answer generation:

### `retriever.py` (Replaces `retrieve_context_node.py`)

This component is responsible for the entire context retrieval pipeline:
1.  Expands the user query.
2.  Performs dense retrieval (semantic search) from ChromaDB.
3.  Performs BM25 lexical retrieval over the same corpus.
4.  Fuses the results using RRF.
5.  Re-ranks the top candidates with FlashRank.
6.  Filters documents based on relevance score.
7.  Selects the final top-N evidence chunks to be passed to the LLM.

### `answer_generator.py` (Replaces `rag_answer_node.py`)

This component takes the processed query and the retrieved evidence chunks to generate a grounded answer:
1.  Builds the final LLM prompt using `prompt_factory.py`.
2.  Invokes the Ollama LLM to generate the answer.
3.  Handles cases of empty questions or LLM generation failures.
4.  Returns the generated answer and associated metadata (e.g., whether context was used, source titles).

## Indexing and Utility Scripts

Supporting scripts are provided for managing the knowledge base and testing the RAG system:

### `scripts/rag_index_local.py`

Indexes local documents (PDF, CSV, TXT, DOCX) into the ChromaDB. It includes functionality to extract metadata like the publication year from filenames.

### `scripts/rag_index_pubmed.py`

Indexes PubMed abstracts by querying the Entrez API. It incorporates crucial filtering mechanisms, such as keyword density and publication type (reviews, guidelines), to ensure only high-quality and relevant medical literature is added to the knowledge base.

### `scripts/rag_index_medlineplus.py`

Indexes MedlinePlus documents into the ChromaDB. It includes functionality to extract metadata and process the documents for retrieval.

### `scripts/check_chroma.py`

A utility script for inspecting the contents of the ChromaDB collection. It can display the total document count, sample documents with their metadata, and perform test searches to verify retrieval.

### `scripts/test_rag_nodes.py`

Provides a comprehensive testing framework for the `retriever` and `answer_generator` components. It supports testing single questions, running predefined batch queries, and an interactive mode, with options for verbose output to aid in debugging.
