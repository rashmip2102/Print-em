import os
import re
import uuid
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, HttpUrl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/render/project/.cache/ms-playwright")
from extractor import extract_clean_article
from ai_generator import generate_full_study_kit
from pdf_compiler import compile_pdf

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Serialize Playwright PDF compiles so concurrent requests don't overload Chromium
COMPILE_SEMAPHORE = asyncio.Semaphore(2)

app = FastAPI(
    title="Study PDF Engine",
    description="Convert educational articles into clean textbook PDFs with AI or offline study kits.",
    version="1.0.0",
)


class GeneratePdfRequest(BaseModel):
    url: HttpUrl
    theme: str = Field(default="textbook", min_length=1, max_length=40)
    gen_summary: bool = True
    gen_saq: bool = True
    gen_laq: bool = True
    gen_mcq: bool = True
    api_key: str | None = None
    model: str | None = None


class GeneratePdfResponse(BaseModel):
    job_id: str
    title: str
    filename: str
    download_url: str
    study_kit_counts: dict[str, int]


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return cleaned[:50] or "study_kit"


def _resolve_llm_config(override_key: str | None = None, override_model: str | None = None):
    raw_key = (
        override_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    api_key = raw_key.strip().strip('"').strip("'") if raw_key else ""

    if override_key:
        if api_key.startswith("gsk_"):
            base_url = "https://api.groq.com/openai/v1"
            default_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        elif api_key.startswith("AQ.") or api_key.startswith("AIza"):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            default_model = "gemini-2.5-flash"
        else:
            base_url = None
            default_model = "gpt-4o-mini"
    elif os.getenv("GEMINI_API_KEY"):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        default_model = "gemini-2.5-flash"
    elif os.getenv("GROQ_API_KEY"):
        base_url = "https://api.groq.com/openai/v1"
        default_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    else:
        base_url = None
        default_model = "gpt-4o-mini"

    model = (override_model or "").strip() or default_model
    return api_key or None, base_url, model


@app.get("/health")
async def health_check():
    api_key, _, model = _resolve_llm_config()
    return {
        "status": "ok",
        "llm_configured": bool(api_key),
        "default_model": model,
    }


@app.post("/api/generate-pdf", response_model=GeneratePdfResponse)
async def create_study_pdf(payload: GeneratePdfRequest):
    job_id = uuid.uuid4().hex[:8]
    target_url = str(payload.url)

    try:
        async with COMPILE_SEMAPHORE:
            extracted = await extract_clean_article(target_url)
            if not extracted or not extracted.get("clean_html"):
                raise HTTPException(status_code=422, detail="Could not extract readable content from URL.")

            api_key, base_url, model = _resolve_llm_config(payload.api_key, payload.model)
            study_kit = await asyncio.to_thread(
                generate_full_study_kit,
                extracted["clean_html"],
                api_key=api_key,
                base_url=base_url,
                model=model,
                gen_summary=payload.gen_summary,
                gen_saq=payload.gen_saq,
                gen_laq=payload.gen_laq,
                gen_mcq=payload.gen_mcq,
            )

            extracted["study_kit"] = study_kit
            extracted["theme"] = payload.theme

            title = extracted.get("title") or "Study Material"
            filename = f"{_slugify(title)}_{job_id}.pdf"
            output_path = OUTPUT_DIR / filename

            await compile_pdf(extracted, str(output_path))

            if not output_path.exists():
                raise HTTPException(status_code=500, detail="PDF file was not created.")

            return GeneratePdfResponse(
                job_id=job_id,
                title=title,
                filename=filename,
                download_url=f"/api/download/{filename}",
                study_kit_counts={
                    "summary": len(study_kit.get("summary", [])),
                    "saq": len(study_kit.get("saq", [])),
                    "laq": len(study_kit.get("laq", [])),
                    "mcqs": len(study_kit.get("mcqs", [])),
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.get("/api/download/{filename}")
async def download_pdf(filename: str):
    # Prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = (OUTPUT_DIR / safe_name).resolve()
    if not file_path.is_relative_to(OUTPUT_DIR.resolve()) or not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=safe_name,
    )


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Print 'em</title>
      <style>
        body { font-family: system-ui, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 16px; color: #1f2937; }
        h1 { margin-bottom: 6px; }
        p.sub { color: #6b7280; margin-top: 0; }
        label { display: block; font-weight: 600; margin-top: 14px; }
        input[type="url"], select { width: 100%; padding: 8px 10px; margin-top: 4px; border: 1px solid #d1d5db; border-radius: 6px; }
        .checks { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
        .checks label { font-weight: 400; margin-top: 0; display: flex; align-items: center; gap: 6px; }
        button { margin-top: 18px; padding: 10px 18px; background: #111827; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        #status { margin-top: 16px; padding: 12px; border-radius: 6px; background: #f3f4f6; display: none; white-space: pre-wrap; }
      </style>
    </head>
    <body>
      <h1>Print 'em</h1>
      <p class="sub">Turn any educational article into a printable study PDF with summary & practice questions.</p>
      <form id="pdf-form">
        <label for="url">Article URL</label>
        <input id="url" name="url" type="url" required placeholder="https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/" />

        <label for="theme">Layout Theme</label>
        <select id="theme" name="theme">
          <option value="textbook" selected>Textbook</option>
          <option value="minimal">Minimal</option>
        </select>

        <div class="checks">
          <label><input type="checkbox" name="gen_summary" checked /> Summary</label>
          <label><input type="checkbox" name="gen_saq" checked /> Short Answers</label>
          <label><input type="checkbox" name="gen_laq" checked /> Long Answers</label>
          <label><input type="checkbox" name="gen_mcq" checked /> MCQs</label>
        </div>

        <button id="submit-btn" type="submit">Generate Study PDF</button>
      </form>
      <div id="status"></div>

      <script>
        const form = document.getElementById("pdf-form");
        const btn = document.getElementById("submit-btn");
        const statusBox = document.getElementById("status");

        form.addEventListener("submit", async (e) => {
          e.preventDefault();
          btn.disabled = true;
          statusBox.style.display = "block";
          statusBox.textContent = "Extracting article, generating study kit, and compiling PDF...";

          const payload = {
            url: document.getElementById("url").value,
            theme: document.getElementById("theme").value,
            gen_summary: form.gen_summary.checked,
            gen_saq: form.gen_saq.checked,
            gen_laq: form.gen_laq.checked,
            gen_mcq: form.gen_mcq.checked,
          };

          try {
            const res = await fetch("/api/generate-pdf", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Request failed");
            statusBox.innerHTML = `Done! <strong>${data.title}</strong><br><a href="${data.download_url}" download>Download ${data.filename}</a>`;
          } catch (err) {
            statusBox.textContent = "Error: " + err.message;
          } finally {
            btn.disabled = false;
          }
        });
      </script>
    </body>
    </html>
    """
