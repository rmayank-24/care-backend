# C.A.R.E. Backend Deployment

This is the backend for the Clinical Assistant for Reasoning and Evaluation (C.A.R.E.) system.

It uses a "chain of chains" RAG (Retrieval-Augmented Generation) approach to provide diagnoses and medication suggestions by querying two separate, specialized Pinecone vector stores:

1.  **`care-mini`**: A knowledge base of diseases, symptoms, and treatments.
2.  **`care-meds`**: A knowledge base of drug and medication information.

## Environment Variables

The following environment variables need to be set in Render (or in a `.env` file for local development):

* `GEMINI_API_KEY`: Your Google Gemini API key.
* `PINECONE_API_KEY`: Your Pinecone API key.

## Deployment

1.  Connect your GitHub repository to Render.
2.  Render will automatically detect the `render.yaml` file (if you have one) or allow you to set up a new Web Service.
3.  Set your environment variables in the Render dashboard.
4.  Deploy the application.

## API Endpoints

### Core Endpoints

* `GET /health`
    * Health check endpoint.
* `POST /api/v1/clinical-analysis`
    * The main endpoint for analyzing unstructured clinical reports.
    * **Body**: `{ "report_text": "..." }`
    * Runs the full "chain of chains" logic:
        1.  Analyzes the report for a diagnosis.
        2.  Searches the medication index based on the diagnosis.
        3.  Returns a combined JSON response.
* `POST /api/v1/analyze-file`
    * Analyzes a clinical report from an uploaded file (PDF or image).
    * **Body**: `multipart/form-data` with a `file` field.
    * Extracts the text and then calls `/api/v1/clinical-analysis`.
* `POST /api/v1/online-resources`
    * Gets online search results for a specific disease name.
    * **Body**: `{ "disease_name": "..." }`

### Analytics Endpoints

* `GET /api/v1/usage-stats`
    * Get usage statistics for the *disease* knowledge base.
* `GET /api/v1/recent-queries`
    * Get a list of recent queries and the *disease* documents they retrieved.

### Debug Endpoints

* `GET /api/v1/debug/vector-store`
    * Checks the status and document count of the **disease** index (`care-mini`).
* `POST /api/v1/debug/retrieval`
    * Tests document retrieval from the **disease** index.
    * **Query Param**: `?query=...`
* `GET /api/v1/debug/vector-store-meds`
    * Checks the status and document count of the **medication** index (`care-meds`).
* `POST /api/v1/debug/retrieval-meds`
    * Tests document retrieval from the **medication** index.
    * **Query Param**: `?query=...`
