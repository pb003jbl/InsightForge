# Overview

A Streamlit-based business intelligence application that combines document analysis, web research, and AI-powered insights to provide strategic business consulting. The application allows users to upload documents, extract insights from web sources, and generate professional consulting-style reports with visualizations.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Single-page web application with interactive components for document upload, query input, and results display
- **Custom Styling**: Professional McKinsey/BCG-inspired CSS with Inter font family and blue gradient color scheme
- **Interactive Components**: File uploaders, text inputs, buttons, and dynamic content rendering

## Backend Architecture
- **Document Processing Pipeline**: Multi-format document ingestion (PDF, web articles) with text extraction and chunking
- **Vector Search System**: FAISS-based similarity search using sentence transformers for semantic document retrieval
- **AI Integration**: Groq API integration with moonshotai/kimi-k2-instruct model for generating strategic business insights
- **Web Research Integration**: Tavily API for real-time web search and trend analysis

## Data Processing
- **Text Extraction**: PyPDF2 for PDF processing, BeautifulSoup and newspaper3k for web content extraction
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 model for generating 384-dimensional document embeddings
- **Chunking Strategy**: Document segmentation for optimal retrieval and context management

## Session Management
- **Streamlit Session State**: Persistent storage for embedder model, FAISS index, and document chunks across user interactions
- **Stateful Components**: Maintains user context and document repository throughout the session

## Visualization
- **Plotly Integration**: Support for interactive charts and graphs (Plotly Express and Graph Objects)
- **Pandas Integration**: Data manipulation and analysis capabilities for structured insights

# External Dependencies

## AI Services
- **Groq API**: Primary LLM service using moonshotai/kimi-k2-instruct model for strategic consulting insights
- **Tavily API**: Web search and trend analysis service for real-time market intelligence

## Machine Learning Libraries
- **SentenceTransformer**: all-MiniLM-L6-v2 model for semantic embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector indexing and retrieval

## Document Processing
- **PyPDF2**: PDF text extraction and processing
- **BeautifulSoup**: HTML parsing and web content extraction
- **newspaper3k**: Article extraction and web scraping

## Visualization and Data
- **Plotly**: Interactive charting and visualization library
- **Pandas**: Data manipulation and analysis framework

## Web Framework
- **Streamlit**: Core web application framework for rapid deployment and interactive UI