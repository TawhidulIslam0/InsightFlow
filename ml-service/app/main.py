import re
from collections import Counter
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = FastAPI(title="InsightFlow ML Service", version="3.0.0")

# 1. Load Hugging Face Zero-Shot Classification pipeline for robust sentiment reasoning
print("Loading Zero-Shot Classification model...")
classifier = pipeline(
    "zero-shot-classification", model="valhalla/distilbart-mnli-12-1"
)

# 2. Load lightweight sentence transformer specifically for semantic recommendations
print("Loading Sentence Transformer for recommendations...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

catalog = [
    (
        "Reflective Journaling",
        "Personal growth",
        "Explore patterns in your thoughts through structured reflection.",
        "reflection personal growth emotions journaling habits goals mindfulness thoughts",
        ["reflection", "growth", "writing"],
    ),
    (
        "Product & UX Research",
        "Research",
        "Turn user feedback into structured product insights and themes.",
        "product users interface feedback experience usability design research customer",
        ["product", "ux", "research"],
    ),
    (
        "Learning Roadmap",
        "Learning",
        "Organize interests and turn them into a practical learning path.",
        "learning education programming skills study course knowledge practice technology",
        ["education", "skills", "planning"],
    ),
    (
        "Creative Project Planning",
        "Projects",
        "Break an idea into milestones, experiments, and measurable outcomes.",
        "project creative idea build plan goals milestones development software design",
        ["projects", "planning", "creative"],
    ),
    (
        "Wellness Habit Tracking",
        "Habits",
        "Use your personal notes to identify recurring routines and habits.",
        "habits routine sleep exercise daily wellness balance energy",
        ["habits", "routine", "wellness"],
    ),
    (
        "Media & Review Analysis",
        "Entertainment",
        "Analyze opinions about movies, games, books, and other media.",
        "movie tv show game book story characters acting music review entertainment",
        ["reviews", "media", "entertainment"],
    ),
]


class Req(BaseModel):
  text: str


def extract_tokens(t: str) -> List[str]:
  stop = set(
      "a an the and or but if then than this that these those is are was were"
      " be been to of in on for with from by as at it its i me my we our you"
      " your they their he she them his her have has had do does did not no"
      " very really just about into over after before can could would should"
      " will what when where who which how".split()
  )
  words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", t.lower())
  return [w for w in words if w not in stop]


def extract_topics(t: str, n: int = 5) -> List[str]:
  tokens = extract_tokens(t)
  ignore_verbs = {
      "transitioning",
      "building",
      "designing",
      "using",
      "making",
      "getting",
      "doing",
      "working",
  }
  filtered = [w for w in tokens if w not in ignore_verbs and len(w) > 3]
  pool = filtered if len(filtered) >= 3 else tokens

  counts = Counter(pool)
  sorted_words = sorted(
      counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True
  )
  return [word for word, _ in sorted_words[:n]] or ["general"]


def generate_summary(t: str, topics: List[str]) -> str:
  sentences = [
      s.strip()
      for s in re.split(r"(?<=[.!?])\s+", t.strip())
      if len(s.strip()) > 10
  ]
  if not sentences:
    return t[:280]

  if len(sentences) == 1:
    sent = sentences[0]
    if len(sent) > 120:
      parts = re.split(r",|but|and|while", sent)
      best_part = max(parts, key=lambda p: len(set(extract_tokens(p))))
      return best_part.strip().capitalize() + "."
    return sent

  scored = []
  for i, sentence in enumerate(sentences):
    sentence_tokens = set(extract_tokens(sentence))
    match_score = len(sentence_tokens & set(topics))
    scored.append((match_score, i, sentence))

  scored.sort(reverse=True, key=lambda x: x[0])
  top_sentences = sorted(scored[:2], key=lambda x: x[1])
  return " ".join(item[2] for item in top_sentences)[:500]


@app.get("/health")
def health():
  return {"status": "ok", "model": "zero-shot-bart-mnli + sentence-transformers"}


@app.post("/analyze")
def analyze(r: Req):
  candidate_labels = ["positive", "negative", "neutral"]
  result = classifier(
      r.text,
      candidate_labels,
      hypothesis_template="This text expresses a {} sentiment.",
  )

  best_sentiment = result["labels"][0].capitalize()
  confidence = round(float(result["scores"][0]), 4)

  topics = extract_topics(r.text)
  summary = generate_summary(r.text, topics)

  return {
      "sentiment": best_sentiment,
      "confidence": confidence,
      "topics": topics,
      "summary": summary,
  }


@app.post("/recommend")
def recommend(r: Req):
  corpus = [item[3] for item in catalog]
  all_texts = corpus + [r.text]

  embeddings = embed_model.encode(all_texts)
  sims = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]
  ranked = np.argsort(sims)[::-1]

  out = []
  for i in ranked[:5]:
    item = catalog[int(i)]
    out.append({
        "title": item[0],
        "category": item[1],
        "description": item[2],
        "score": round(float(sims[int(i)]), 4),
        "tags": item[4],
    })

  return {"recommendations": out}