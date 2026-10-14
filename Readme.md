# 💻 Laptop Advisor

A RAG-based assistant that recommends laptops for the Indian market. Describe your budget and needs in plain language, and it retrieves matching laptops from a vector database and explains its recommendations with an LLM.

**Live demo:** [YOUR_CLOUD_RUN_URL](YOUR_CLOUD_RUN_URL)
*(Hosted on Google Cloud Run. The first load after idle can take 10-30 seconds because of cold starts.)*

![Screenshot](docs/screenshot.png)

---

## How it works

```
User question
     │
     ▼
Streamlit UI ──► Embed query (fastembed)
                      │
                      ▼
              Qdrant vector search ──► top matching laptops
                      │
                      ▼
          Groq LLM (answers only from retrieved context)
                      │
                      ▼
              Recommendation + reasoning
```

1. **Ingest:** laptop data from a CSV is cleaned, embedded, and stored in Qdrant.
2. **Retrieve:** the user's query is embedded and matched against the stored laptops.
3. **Generate:** the LLM answers using only the retrieved laptops, and says so when the data doesn't cover the question.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Vector database | Qdrant Cloud |
| Embeddings | fastembed |
| LLM | Groq |
| Packaging | Docker |
| Hosting | Google Cloud Run |

## Project structure

```
.
├── app.py            # Streamlit app
├── core/             # Retrieval and generation logic
├── ingest/           # Data loading and embedding scripts
├── data/             # Laptop dataset
├── tests/
├── Dockerfile
└── requirements.txt
```

## Run locally

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/Laptop-Advisor.git
cd Laptop-Advisor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file (never commit it):
```
QDRANT_URL=https://your-cluster-url:6333
QDRANT_API_KEY=your-qdrant-key
GROQ_API_KEY=your-groq-key
```

**3. Load the data into Qdrant** (run the script in `ingest/`)

**4. Start the app**
```bash
streamlit run app.py
```
Open http://localhost:8501.

## Run with Docker

```bash
docker build -t laptop-advisor .
docker run -p 8080:8080 \
  -e QDRANT_URL="https://your-cluster-url:6333" \
  -e QDRANT_API_KEY="your-qdrant-key" \
  -e GROQ_API_KEY="your-groq-key" \
  laptop-advisor
```
Open http://localhost:8080.

## Deploy to Google Cloud Run

```bash
gcloud run deploy laptop-advisor \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --env-vars-file env.yaml
```

`env.yaml` holds the same three variables as above in YAML form. It is listed in `.gitignore` and `.dockerignore`, so keys never end up in the image or the repository.

## What I learned

- Packaging a Python app with Docker, including matching the Python version and keeping the build context small with `.dockerignore`
- Deploying to Cloud Run and fixing IAM permission errors for source deploys
- Keeping API keys out of images and Git

## Limitations

- The dataset is a static snapshot, so prices and availability may be out of date.
- Recommendations are limited to the laptops in the dataset.

## Roadmap (v2)

- [ ] LangGraph agent with query understanding, validation, and fallbacks
- [ ] Redis caching to cut latency and LLM calls
- [ ] Scheduled pipeline to refresh the RAG data
- [ ] Live price and stock data through a source-adapter layer
- [ ] Evaluation set to measure retrieval and answer quality

## License

Add a license of your choice (for example MIT).