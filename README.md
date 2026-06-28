# Fixate

Fixate is an AI growth simulator for marketing pages and campaign assets. It lets you test a URL or pasted HTML before spending on traffic, then returns attention analysis, buyer-response scoring, conversion blockers, improved variants, edited images, and an A/B test plan.

The app uses OpenAI as the only external AI service:

- OpenAI vision predicts attention/fixation regions for the heatmap.
- OpenAI vision scores buyer-response signals, zones, blockers, and Fixate Score.
- OpenAI agents generate buyer reactions, strategy, creative variants, and A/B plans.
- OpenAI image editing creates edited variant images.

Local code handles Playwright capture, file storage, SSE streaming, and serving generated artifacts.

## Project Structure

```text
backend/   FastAPI API and optimization pipeline
frontend/  React + Vite UI
```

## Environment

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_IMAGE_MODEL=gpt-image-1
FIXATE_JOBS_DIR=
```

`FIXATE_JOBS_DIR` is optional. Leave it blank to store screenshots, heatmaps, and edited images in your system temp folder.

## Run Locally

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
uvicorn main:app --reload --port 8080
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## API Smoke Test

Health:

```bash
curl http://127.0.0.1:8080/health
```

Start an optimization job:

```bash
JOB_ID=$(curl -s -X POST http://127.0.0.1:8080/optimize \
  -H 'Content-Type: application/json' \
  -d '{
    "url":"https://example.com",
    "target_customer":"startup founder",
    "goal":"increase signups",
    "iterations":1,
    "constraints":{
      "brand":{"colors":["#0D7D59"],"fonts":["Inter"],"tone":"clear","logo_present":false},
      "locked_elements":[],
      "aggressiveness":"balanced"
    }
  }' | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')

curl -N "http://127.0.0.1:8080/job/$JOB_ID/stream"
curl -s "http://127.0.0.1:8080/job/$JOB_ID/result" | python3 -m json.tool
open "http://127.0.0.1:8080/job/$JOB_ID/heatmap"
open "http://127.0.0.1:8080/job/$JOB_ID/variant/1/image"
```

## Main Endpoints

- `GET /health`
- `POST /capture`
- `GET /capture/{capture_id}/image`
- `GET /capture/{capture_id}/heatmap`
- `POST /score/{capture_id}`
- `POST /optimize`
- `GET /job/{job_id}/stream`
- `GET /job/{job_id}/result`
- `GET /job/{job_id}/heatmap`
- `GET /job/{job_id}/variant/{iteration}/image`
- `GET /job/{job_id}/best-image`

## Notes

- A run can take 30-90 seconds because OpenAI image editing is slower than text calls.
- If the OpenAI API fails, the backend has local fallbacks so development does not crash.
- Do not commit `backend/.env`; it contains your API key.
