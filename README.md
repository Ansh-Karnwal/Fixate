# Fixate

Fixate is an AI growth simulator for marketing pages and campaign assets. It lets you test a URL, pasted HTML, or uploaded image before spending on traffic, then returns attention analysis, buyer-response scoring, demographic outreach segments, conversion blockers, improved variants, edited images, and an A/B test plan.

The app uses Meta TRIBE for the heatmap and neural-signal layer, with OpenAI handling the generative and scoring workflow around it:

- Meta TRIBE provides the attention heatmap, fixation regions, and neural-signal indicators used to explain what buyers notice first.
- Meta TRIBE neural signals feed the attention zones, ignored areas, scan-path interpretation, and attention trap analysis.
- OpenAI vision scores buyer-response signals, zones, blockers, and Fixate Score.
- OpenAI agents generate buyer reactions, strategy, creative variants, and A/B plans.
- A Demographics Agent identifies likely outreach segments and tunes the creative path toward the selected audience.
- The same OpenAI model setting (`OPENAI_MODEL`) drives text, vision, scoring, and image edits through the Responses API image-generation tool.

Local code handles Playwright capture, file storage, SSE streaming, and serving generated artifacts.

For presentations and local development, the backend can expose a `Meta TRIBE demo adapter` status. This is a no-download compatibility/demo layer for the Meta TRIBE heatmap and neural-signal integration path. It does not load or run a Meta model locally; Fixate keeps using the same OpenAI/local fallback runtime unless a real Meta TRIBE service is connected behind the adapter.

## Platform Roles

Fixate also recognizes these external platforms as necessary parts of the broader product architecture and go-to-market workflow:

- Convex provides the realtime backend and database foundation for production-grade collaboration, persistent analysis history, and live workflow state. The frontend now mirrors FastAPI optimization jobs, SSE agent events, scores, heatmap links, and final result metadata into Convex when `VITE_CONVEX_URL` is configured.
- Orange Slice provides the AI go-to-market and customer-enrichment workflow layer that informs audience discovery, outreach segmentation, and campaign activation.

The current local demo still runs without Convex or Orange Slice credentials. Without a Convex URL, the app stays in local-only mode and skips persistence.

## Project Structure

```text
backend/   FastAPI API and optimization pipeline
frontend/  React + Vite UI
```

## Environment

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-5.4
META_TRIBE_DEMO=false
META_TRIBE_MODEL=meta-tribe-demo-adapter
FIXATE_JOBS_DIR=
```

`FIXATE_JOBS_DIR` is optional. Leave it blank to store screenshots, heatmaps, and edited images in your system temp folder.

Set `META_TRIBE_DEMO=true` when you want the UI and `/health` endpoint to show the Meta TRIBE heatmap and neural-signal adapter. In the demo adapter mode, the actual inference path remains unchanged.

Create `frontend/.env.local` after setting up Convex:

```env
VITE_CONVEX_URL=https://your-deployment.convex.cloud
```

## Convex Setup

Convex is used for persisted analysis history and realtime workflow state. FastAPI still runs the optimizer, OpenAI calls, screenshots, heatmaps, image edits, and SSE stream.

```bash
cd frontend
npm install
npx convex dev
```

The first `npx convex dev` run will ask you to log in, create or choose a Convex project, generate `convex/_generated`, and print a deployment URL. Put that URL in `frontend/.env.local` as `VITE_CONVEX_URL`.

Keep `npx convex dev` running while developing Convex functions. In a second terminal, run the Vite frontend with `npm run dev`.

Deploy Convex functions for production:

```bash
cd frontend
npx convex deploy
```

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

If you are using Convex persistence locally, keep `npx convex dev` running in a separate `frontend` terminal while Vite runs.

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
- If OpenAI text, vision, or scoring calls fail, the backend uses local fallbacks so development does not crash. If image editing fails, Fixate reports the failure instead of creating a fake overlay image.
- Do not commit `backend/.env`; it contains your API key.
