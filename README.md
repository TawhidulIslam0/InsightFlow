# InsightFlow — AI-Powered Personal Data & Recommendation Platform

A complete full-stack application for analyzing personal text with machine learning, storing results, and generating similarity-based recommendations.

## Stack
- React + TypeScript + Vite
- Node.js + Express + TypeScript
- PostgreSQL + Prisma
- Python + FastAPI + scikit-learn
- JWT + bcrypt
- Docker Compose

## Features
- Registration and login
- Protected JWT API
- Personal dashboard
- Text sentiment classification
- Confidence scoring
- Topic extraction
- Extractive AI summary
- Similarity-based recommendations
- Analysis history and deletion
- Responsive dark/light UI
- Separate frontend, backend, database, and ML service

## Docker
```bash
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose up --build
```
Open http://localhost:5173. Backend is on http://localhost:4000 and ML docs are at http://localhost:8000/docs.

## Local development
Create a PostgreSQL database named `insightflow`.

Backend:
```bash
cd backend
npm install
npx prisma generate
npx prisma migrate dev --name init
npm run dev
```

ML:
```bash
cd ml-service
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## ML implementation
The local ML service uses TF-IDF features and logistic regression for sentiment classification, keyword extraction for topics, extractive sentence selection for summaries, and TF-IDF cosine similarity against a recommendation catalog. It requires no paid AI API and can later be replaced with transformer embeddings or an external LLM.
