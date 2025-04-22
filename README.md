# Binge 🍽️

**Binge** is a full‑stack, swipe‑based restaurant recommendation platform.  Think “Tinder for food”.

The stack is split into a **FastAPI + MongoDB backend** (Python) and a **Vite + React frontend** (JavaScript/TypeScript‑ready).  All data is stored in MongoDB via Beanie ODM, and the recommendation engine combines

- Sentence‑Transformer attribute embeddings ✨
- Approximate nearest neighbour search with **Annoy**
- Geospatial filtering with a fast Haversine implementation

---

## Features

| Layer           | Highlights                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| **Backend**     | FastAPI ★ JWT auth ★ Beanie ODM ★ Robust routers for users, auth, restaurants, photos & recommendations      |
| **Recommender** | Attribute‑aware content embeddings · Rating, distance & price signals                                        |
| **Frontend**    | Vite + React 18 · MUI v5 · Tinder‑Card interactions · Auth context with JWT · Profile & add‑restaurant flows |

---

## Directory Layout

```text
./
├── backend/        ← FastAPI app
│   ├── models/     ← Pydantic/Beanie documents
│   ├── routers/    ← API routes (auth, photos, …)
│   ├── recommendations/
│   │   ├── data/               ← Yelp & Michelin JSON/CSV dumps
│   │   └── model/              ← Core recommender code
│   │       ├── preprocess.ipynb
│   │       ├── recommendation_model.py
│   │       └── populatedb.py
│   ├── database.py ← Mongo initialization
│   └── main.py     ← FastAPI entry‑point
└── frontend/
    ├── public/
    │   └── photos/  ← Yelp images (<photo_id>.jpg)
    └── src/        ← components, context, pages
```

---

## Prerequisites

| Tool                    | Version (tested)    | Notes                      |
| ----------------------- | ------------------- | -------------------------- |
| **Python**              | 3.10 – 3.12         | Used by FastAPI backend    |
| **Node** & **npm**      | Node ≥ 18 / npm ≥ 9 | Vite frontend              |
| **MongoDB**             | 6.x                 | Replica‑set *not* required |
| **Poetry** *or* **pip** | latest              | Dependency management      |

---

## 1 · Clone & configure

```bash
$ git clone https://github.com/mosmolov/binge.git
$ cd binge
```

Create **.env** files at project root ***and*** inside `/backend` with:

```env
MONGO_URI=
MONGO_DB_NAME=
JWT_SECRET_KEY=ChangeMeNow
```

> If you keep a single `.env` in the repo root, Beanie/routers will still load it because `python-dotenv` walks up the tree.

---

## 2 · Backend setup

```bash
# (a) Create virtual env – pick one:
python -m venv .venv          # stdlib
# or
poetry shell                  # if you use Poetry

# (b) Install deps
pip install -r backend/requirements.txt   # if you use requirements
#   – OR –
poetry install

# (c) Populate sample data (optional – needs Yelp/Michelin dumps)
python backend/recommendations/model/populatedb.py

# (d) Run API server (auto‑reload for dev)
uvicorn backend.main:app --reload --port 8000
```

### Useful dev URLs

- Swagger UI:         `http://localhost:8000/docs`
- ReDoc:              `http://localhost:8000/redoc`
- Health check:       `GET /recommendations/health`

---

## 3 · Frontend setup

```bash
cd frontend
npm install          # or pnpm / yarn

# Environment (optional ­– default API is localhost:8000)
#   VITE_API_BASE=http://localhost:8000

npm run dev          # open http://localhost:5173
```

The React app will hot‑reload and talk to the backend API via the `axios.defaults.baseURL` defined in `src/context/AuthContext.jsx`.

---

## Environment Variables (backend)

| Var              | Required | Default            | Description               |
| ---------------- | -------- | ------------------ | ------------------------- |
| `MONGO_URI`      | ✅        | —                  | MongoDB connection string |
| `MONGO_DB_NAME`  | ✅        | —                  | Database name             |
| `JWT_SECRET_KEY` | ✅        | —                  | Secret for HS256 JWTs     |
| `SENTENCE_MODEL` | optional | `all-MiniLM-L6-v2` | SBERT model name (HF hub) |

---

---

## 4 · Download required datasets

The recommender needs two public datasets before you populate MongoDB:

| Dataset                             | Where to grab it                                                                                                                                       | What to copy                          | Destination                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------- |
| **Yelp Open Dataset (photos)**      | [https://www.yelp.com/dataset](https://www.yelp.com/dataset)                                                                                           | Everything inside `photos/`           | `frontend/public/photos/`       |
| **Yelp Open Dataset (businesses)**  | same link                                                                                                                                              | `yelp_academic_business_dataset.json` | `backend/recommendations/data/` |
| **Michelin Guide Restaurants 2021** | [https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021](https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021) | CSV file(s) from the download         | `backend/recommendations/data/` |

> **Important:** The backend assumes photos are saved as `<photo_id>.jpg` inside **`frontend/public/photos/`**.  After downloading, you can delete any unneeded Yelp tables to save space.
