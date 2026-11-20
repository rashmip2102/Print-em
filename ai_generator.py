import os
import re
from bs4 import BeautifulSoup
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from google.genai import types
import httpx

class ShortQuestion(BaseModel):
    question: str
    sample_answer: str

class LongQuestion(BaseModel):
    question: str
    detailed_solution: str

class MCQ(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str

class FullStudyKit(BaseModel):
    summary_bullets: list[str]
    short_questions: list[ShortQuestion]
    long_questions: list[LongQuestion]
    mcqs: list[MCQ]

def _clean_math_text(text: str) -> str:
    if not text:
        return text
    s = (
        text.replace(r"\_", "_")
        .replace(r"\frac", "frac")
        .replace(r"\sigma", "σ")
        .replace(r"\approx", "≈")
    )
    # Unwrap $...$ expressions cleanly and strip any remaining stray $
    s = re.sub(r"\$([^$\n]+)\$", r"\1", s)
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _generate_offline_study_kit(soup: BeautifulSoup) -> FullStudyKit:
    """Deterministic local NLP extractor used when offline or without an API key."""
    term_defs = []
    for li in soup.find_all("li"):
        strong = li.find(["strong", "b"])
        full_txt = li.get_text(" ", strip=True)
        if strong:
            term = strong.get_text(" ", strip=True).rstrip(":")
            defn = full_txt.replace(strong.get_text(" ", strip=True), "", 1).lstrip(": -–—").strip()
            if 2 <= len(term) <= 60 and len(defn) >= 25:
                term_defs.append((term, defn))

    sections = []
    for heading in soup.find_all(["h2", "h3"]):
        h_text = heading.get_text(" ", strip=True)
        if len(h_text) < 4 or any(w in h_text.lower() for w in ("table of contents", "reference", "conclusion")):
            continue
        collected = []
        for sib in heading.find_next_siblings():
            if sib.name in ("h2", "h3"):
                break
            if sib.name in ("p", "ul", "ol"):
                txt = sib.get_text(" ", strip=True)
                if len(txt) > 30:
                    collected.append(txt)
            if len(" ".join(collected)) > 450:
                break
        if collected:
            sections.append((h_text, " ".join(collected)[:550]))

    summary = [f"{t}: {d}" for t, d in term_defs[:6]]
    if len(summary) < 4:
        for p in soup.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if len(txt) > 60 and txt not in summary:
                summary.append(txt[:220] + ("..." if len(txt) > 220 else ""))
            if len(summary) >= 5:
                break

    saqs = []
    for term, defn in term_defs[:3]:
        saqs.append(ShortQuestion(
            question=f"Define '{term}' and state its role in this topic.",
            sample_answer=defn
        ))
    if not saqs:
        for h_text, body in sections[:3]:
            saqs.append(ShortQuestion(
                question=f"Briefly explain {h_text}.",
                sample_answer=body[:250]
            ))

    laqs = []
    for h_text, body in sections[:2]:
        laqs.append(LongQuestion(
            question=f"Explain {h_text} in detail with key steps or components.",
            detailed_solution=body
        ))

    mcqs = []
    pool = term_defs if len(term_defs) >= 4 else [(h, b[:140]) for h, b in sections]
    letters = ["A", "B", "C", "D"]
    for idx in range(min(4, len(pool))):
        target_term, target_def = pool[idx]
        distractors = [pool[j][1][:110] for j in range(len(pool)) if j != idx][:3]
        while len(distractors) < 3:
            distractors.append("None of the above mechanisms apply in this context.")

        raw_choices = [target_def[:110]] + distractors
        shift = idx % 4
        rotated = raw_choices[-shift:] + raw_choices[:-shift] if shift else raw_choices
        correct_idx = rotated.index(target_def[:110])

        formatted_opts = [f"{letters[i]}) {opt}" for i, opt in enumerate(rotated)]
        mcqs.append(MCQ(
            question=f"Which of the following best describes {target_term}?",
            options=formatted_opts,
            correct_answer=formatted_opts[correct_idx],
            explanation=f"{target_term}: {target_def}"
        ))

    return FullStudyKit(
        summary_bullets=summary or ["Key concepts extracted from the source article."],
        short_questions=saqs,
        long_questions=laqs,
        mcqs=mcqs
    )

def generate_full_study_kit(
    article_html: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4o-mini",
    gen_summary: bool = True,
    gen_saq: bool = True,
    gen_laq: bool = True,
    gen_mcq: bool = True,
) -> dict:
    soup = BeautifulSoup(article_html, "html.parser")
    clean_key = api_key.strip().strip('"').strip("'") if api_key else ""

    if clean_key:
        try:
            clean_text = soup.get_text(separator="\n", strip=True)
            system_prompt = """
            You are an expert Computer Science professor creating printable study material.
            Generate content grounded STRICTLY in the provided article text. Do not invent outside facts.
            Respond ONLY with raw valid JSON (no markdown code fences, no extra commentary) matching this exact schema:
            {
              "summary_bullets": ["point 1", "point 2", "... (5 to 7 concise takeaways)"],
              "short_questions": [
                {"question": "...", "sample_answer": "..."}
              ],
              "long_questions": [
                {"question": "...", "detailed_solution": "..."}
              ],
              "mcqs": [
                {
                  "question": "...",
                  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                  "correct_answer": "A) ...",
                  "explanation": "..."
                }
              ]
            }
            Include 3 short_questions, 2 long_questions, and 5 mcqs.
            """

            if clean_key.startswith("AQ."):
                genai_client = genai.Client(api_key=clean_key)
                gemini_model = model if model and "gemini" in model else "gemini-2.5-flash"
                response = genai_client.models.generate_content(
                    model=gemini_model,
                    contents=f"ARTICLE CONTENT:\n{clean_text[:12000]}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.3,
                        response_mime_type="application/json",
                    ),
                )
                raw_content = response.text or ""
            else:
                with httpx.Client(
                    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                    timeout=30.0,
                ) as http_client:
                    client = OpenAI(
                        api_key=clean_key,
                        base_url=base_url,
                        http_client=http_client,
                    )
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"ARTICLE CONTENT:\n{clean_text[:12000]}"},
                        ],
                        temperature=0.3,
                    )
                    raw_content = response.choices[0].message.content or ""

            cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.IGNORECASE)
            validated_kit = FullStudyKit.model_validate_json(cleaned_json)
        except Exception as e:
            detail = getattr(e, "body", None) or str(e)
            print(f"API call failed ({detail}) — switching to offline extractor.")
            validated_kit = _generate_offline_study_kit(soup)
    else:
        validated_kit = _generate_offline_study_kit(soup)

    summary_out = [_clean_math_text(item) for item in validated_kit.summary_bullets] if gen_summary else []
    saq_out = [
        {
            "question": _clean_math_text(q.question),
            "sample_answer": _clean_math_text(q.sample_answer),
        }
        for q in validated_kit.short_questions
    ] if gen_saq else []
    laq_out = [
        {
            "question": _clean_math_text(q.question),
            "detailed_solution": _clean_math_text(q.detailed_solution),
        }
        for q in validated_kit.long_questions
    ] if gen_laq else []
    mcq_out = [
        {
            "question": _clean_math_text(m.question),
            "options": [_clean_math_text(opt) for opt in m.options],
            "correct_answer": _clean_math_text(m.correct_answer),
            "explanation": _clean_math_text(m.explanation),
        }
        for m in validated_kit.mcqs
    ] if gen_mcq else []

    return {
        "summary": summary_out,
        "saq": saq_out,
        "laq": laq_out,
        "mcqs": mcq_out,
    }

if __name__ == "__main__":
    with open("extracted_raw.html", "r", encoding="utf-8") as f:
        sample_html = f.read()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if os.getenv("GEMINI_API_KEY"):
        base_url, model = "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"
    elif os.getenv("GROQ_API_KEY"):
        base_url, model = "https://api.groq.com/openai/v1", os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    else:
        base_url, model = None, "gpt-4o-mini"

    kit = generate_full_study_kit(sample_html, api_key=api_key, base_url=base_url, model=model)
    print(f"Generated {len(kit['summary'])} summary points, {len(kit['saq'])} SAQs, {len(kit['laq'])} LAQs, {len(kit['mcqs'])} MCQs.")