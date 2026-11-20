import os
import html
import asyncio
from jinja2 import Environment, FileSystemLoader
from playwright.async_api import async_playwright
from extractor import extract_clean_article
from ai_generator import generate_full_study_kit

async def compile_pdf(data: dict, output_filepath: str):
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.jinja2")

    rendered_html = template.render(
        title=data["title"],
        source_url=data["source_url"],
        clean_html=data["clean_html"],
        theme=data.get("theme", "textbook"),
        study_kit=data.get("study_kit")
    )

    safe_title = html.escape(data["title"][:60])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.set_content(rendered_html, wait_until="networkidle", timeout=20000)
        except Exception:
            await page.wait_for_timeout(1000)

        await page.pdf(
            path=output_filepath,
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=f"""
                <div style="font-size: 9px; font-family: Arial, sans-serif; color: #666; width: 100%; padding: 0 15mm; display: flex; justify-content: space-between;">
                    <span>{safe_title}</span>
                    <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
                </div>
            """
        )
        await browser.close()

    print(f"PDF compiled successfully: {os.path.abspath(output_filepath)}")

if __name__ == "__main__":
    async def main():
        test_url = "https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/"
        extracted = await extract_clean_article(test_url)

        # Pick whichever API key you have set in your terminal:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

        if os.getenv("GEMINI_API_KEY"):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            model = "gemini-2.5-flash"
        elif os.getenv("GROQ_API_KEY"):
            base_url = "https://api.groq.com/openai/v1"
            model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        else:
            base_url = None
            model = "gpt-4o-mini"

        if api_key:
            print(f"Generating AI Study Kit using {model}...")
            extracted["study_kit"] = generate_full_study_kit(
                extracted["clean_html"],
                api_key=api_key.strip(),
                base_url=base_url,
                model=model
            )
        else:
            print("No API key found — compiling article without AI study kit.")

        await compile_pdf(extracted, "outputs/neural_networks_study_kit.pdf")

    asyncio.run(main())