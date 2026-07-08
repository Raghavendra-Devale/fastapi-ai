# FastAPI AI Platform

A backend service built with FastAPI, integrated with AI features including embeddings, vector search (using pgvector), recommendations, and task workers. It acts as the machine learning engine for the Job Platform, parsing resumes and generating cosine similarity matches and explanations.

---

## Project Structure

```
fastapi-ai/
├── app/
│   ├── api/
│   │   ├── health/             # Health API controllers
│   │   ├── recommendations/    # Recommendations API controllers
│   │   └── v1/
│   │       ├── endpoints/      # API endpoints (resume parsing)
│   │       └── router.py       # API Router
│   ├── application/            # Application use-case orchestrators
│   │   ├── ai/                 # App-level AI embeddings orchestrator
│   │   └── resume/             # App-level resume processing orchestrator
│   ├── core/                   # Configuration, exceptions, logging, middleware
│   ├── domain/                 # Domain entities, services and logic boundaries
│   │   ├── ai/                 # LLM/Embedding provider interfaces & models
│   │   ├── jobs/               # Domain job model & parser types
│   │   ├── recommendation/     # Cosine scoring & match explanations
│   │   └── resume/             # Domain PDF extractor & text normalizer
│   ├── utils/                  # Helper & utility functions
│   └── main.py                 # Application entry point & lifecycle hooks
├── tests/              # Pytest test suite
├── .env                # Local environment secrets
├── .env.example        # Reference environment variables
├── Dockerfile          # Container setup
├── docker-compose.yml  # Local orchestration (App + PostgreSQL with pgvector)
├── pyproject.toml      # Poetry/uv dependency specs
└── README.md           # Documentation
```

---

## 🧭 API Endpoints

### 1. Resume Parsing
- **Endpoint:** `POST /api/v1/resume/process`
- **Method:** `Multipart Form Upload`
- **Logic:** Reads the uploaded PDF file using `PyMuPDF` (`fitz`), extracts raw text, and employs structured parsing to output structured JSON:
  - Summaries
  - Technical skills (with confidence ratings)
  - Work experiences (roles, companies, dates, responsibilities)
  - Education (degrees, schools, dates)
  - Projects (names, descriptions, tech stack)

### 2. Job Recommendations Matcher
- **Endpoint:** `POST /api/v1/recommendations/generate`
- **Method:** `JSON Request`
- **Logic:**
  1. Computes vector embeddings for the candidate profile and job listings using the `all-MiniLM-L6-v2` transformer model.
  2. Runs cosine similarity matching to rank jobs.
  3. Uses a local **Ollama** LLM connection (e.g., `llama3` model) to construct contextual explanation strings detailing why each recommended job matches the candidate's skill set.

### 3. Service Health Checks
- **Endpoint:** `GET /api/v1/health`
- **Logic:** Performs diagnostic checks against dependencies, reporting if the Ollama provider is reachable and active.

---

## Setup & Running

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Environment
```bash
cp .env.example .env
```

To swap base connection URLs, edit `.env`:
- **Local host execution:** Set `AI_OLLAMA_BASE_URL=http://localhost:11434`
- **Docker Compose execution:** Set `AI_OLLAMA_BASE_URL=http://host.docker.internal:11434`

### 3. Start with Docker Compose
```bash
docker compose up --build
```

### 4. Access the API
- API Endpoint: http://localhost:8000/
- API Docs (Swagger UI): http://localhost:8000/docs

---

## Configuration

The application is configured using environment variables prefixed with `AI_`. See `.env.example` for details.

### Key Environment Variables
*   `AI_PROVIDER`: The selected AI provider. Currently supports `ollama`.
*   `AI_OLLAMA_BASE_URL`: Base connection URL of the running Ollama API service (e.g. `http://localhost:11434`).
*   `AI_EMBEDDING_MODEL`: Configured model used for generating embeddings.
*   `AI_LLM_MODEL`: Configured LLM model used for chat completions.

### Startup Validation
On application boot, the engine automatically resolves the configured AI provider and runs a startup validation check calling its health check API. If the provider is unreachable, a warning is printed to the structured logs, but the application continues to start up.

### Health Endpoint
A detailed health check is available at:
*   `GET /api/v1/health`

This endpoint returns HTTP 200 health status including nested dependency states even if third-party providers (like Ollama) are offline:
```json
{
  "status": "UP",
  "service": "fastapi-ai",
  "version": "1.0.0",
  "environment": "development",
  "dependencies": {
    "provider": {
      "healthy": true,
      "provider": "ollama",
      "message": "Ollama service is reachable and healthy."
    }
  }
}
```
