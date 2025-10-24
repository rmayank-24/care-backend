# C.A.R.E. Backend Deployment

This is the backend for the Clinical Assistant for Reasoning and Evaluation (C.A.R.E.) system.

## Environment Variables

The following environment variables need to be set in Render:

- `GEMINI_API_KEY`: Your Google Gemini API key
- `PINECONE_API_KEY`: Your Pinecone API key

## Deployment

1. Connect your GitHub repository to Render
2. Render will automatically detect the render.yaml file
3. Set your environment variables in the Render dashboard
4. Deploy the application

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/v1/analyze` - Analyze clinical reports
- `POST /api/v1/clinical-analysis` - Get structured clinical analysis
- `POST /api/v1/online-resources` - Get online resources for a disease
- `GET /api/v1/usage-stats` - Get usage statistics
- `GET /api/v1/recent-queries` - Get recent queries
- `GET /api/v1/debug/vector-store` - Debug vector store
- `POST /api/v1/debug/retrieval` - Debug document retrieval
- `POST /api/v1/debug/rag-chain` - Debug RAG chain