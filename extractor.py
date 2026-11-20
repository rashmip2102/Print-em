import re
import html
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

def render_tex_to_html(html_str: str) -> str:
    def _convert(match):
        expr = match.group(1).strip()
        # Unescape GFG's double-escaped LaTeX artifacts
        expr = expr.replace(r"\backslash ", "\\").replace(r"\backslash", "\\")
        expr = expr.replace(r"\{", "{").replace(r"\}", "}")
        expr = expr.replace(r"^{\wedge}", "^").replace(r"\wedge", "^")

        # Convert fractions: \frac{a}{b} -> (a) / (b)
        expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", expr)

        # Replace common LaTeX symbols with crisp Unicode
        symbols = {
            r"\sigma": "σ", r"\Sigma": "Σ", r"\alpha": "α", r"\beta": "β",
            r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ", r"\pi": "π",
            r"\approx": "≈", r"\times": "×", r"\cdot": "·", r"\ldots": "…",
            r"\dots": "…", r"\leq": "≤", r"\geq": "≥", r"\neq": "≠",
            r"\rightarrow": "→", r"\infty": "∞", r"\sum": "∑",
        }
        for tex_cmd, uni in symbols.items():
            expr = expr.replace(tex_cmd, uni)

        expr = html.escape(expr)
        # Convert subscripts and superscripts to HTML tags
        expr = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", expr)
        expr = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", expr)
        expr = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", expr)
        expr = re.sub(r"\^([A-Za-z0-9\-]+)", r"<sup>\1</sup>", expr)

        return f'<span class="math-inline">{expr}</span>'

    return re.sub(
        r"\[?\s*\$?\s*\[Tex\](.*?)\[/Tex\]\s*\$?\s*\]?",
        _convert,
        html_str,
        flags=re.DOTALL | re.IGNORECASE,
    )

async def extract_clean_article(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = await context.new_page()

        blocked_domains = (
            "googlesyndication", "doubleclick", "google-analytics",
            "googletagmanager", "facebook", "taboola", "adservice", "scorecardresearch"
        )
        await page.route(
            "**/*",
            lambda route: route.abort() if any(d in route.request.url for d in blocked_domains) else route.continue_()
        )

        response = await page.goto(url, wait_until="commit", timeout=45000)
        if response and response.status >= 400:
            await browser.close()
            raise ValueError(f"Page returned HTTP {response.status}. Check that the URL is valid.")

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=20000)
        except Exception:
            pass

        # Wait for a visible heading/main container rather than the first hidden nav <h1>
        try:
            await page.locator(
                "main h1:visible, article h1:visible, #main h1:visible, h1:visible, main, article"
            ).first.wait_for(state="visible", timeout=10000)
        except Exception:
            pass

        await page.wait_for_timeout(1500)

        await page.evaluate("""
            async () => {
                await new Promise((resolve) => {
                    let totalDistance = 0;
                    const distance = 600;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalDistance += distance;
                        if (totalDistance >= scrollHeight) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, 60);
                });
            }
        """)
        await page.wait_for_timeout(1500)

        content = await page.content()
        await browser.close()

    soup = BeautifulSoup(content, "html.parser")

    # 1. Remove boilerplate, ads, hidden SEO transcripts, and carousel UI
    for selector in [
        "header", "footer", "nav", "aside", "button", ".sidebar", ".ad-container",
        "#comments", ".utility-bar", ".header-wrapper", ".footer-wrapper",
        ".carousel", ".slider", ".swiper", ".slick-slider",
        ".carousel-control-prev", ".carousel-control-next", ".slick-arrow",
        ".carousel-indicators", ".slick-dots", ".swiper-pagination",
        "[class*='carousel']", "[class*='slider']",
        "aside",
        ".modal", ".lightbox", "[aria-hidden='true']", ".sr-only", ".visually-hidden",
        "[hidden]", "[style*='display: none']", "[style*='display:none']",
        "[style*='visibility: hidden']", "[style*='visibility:hidden']",
        "[style*='opacity: 0']", "[style*='opacity:0']",
        "script", "style", "iframe", ".gfg-burr-ad", ".cookie-consent",
        ".rightBar", ".leftBar", ".article_bottom_text",
    ]:
        for tag in soup.select(selector):
            if tag.parent is not None:
                tag.decompose()

    article_body = (
     soup.find("div", class_="article--viewer_content")
     or soup.find("article")
     or soup.find("div", class_="entry-content")
     or soup.find(id="main")
     or soup.find("div", class_="w3-main")
     or soup.find("main")
     or soup.body
    )

    # 2. Remove carousel counters ("1/4"), pause controls ("< II >"), and hidden diagram OCR text
    for tag in list(article_body.find_all(["span", "div", "i", "p"])):
        if tag.parent is None or tag.find("img"):
            continue
        txt = tag.get_text(strip=True)
        if re.fullmatch(r"(\d+\s*/\s*\d+|<|>|\|\||II|▶|◀|►|◄|<\s*(II|\|\||▶|◀|►|◄)\s*>)", txt):
            tag.decompose()
            continue
        # Strip GFG's hidden image-transcript blocks placed right inside/beside image containers
        classes = " ".join(tag.get("class", [])).lower()
        if any(k in classes for k in ("transcript", "ocr", "caption-hidden", "carousel-control", "slider-control")):
            tag.decompose()
            continue

    # 3. Normalize all GFG code containers into clean <pre><code> blocks
    code_candidates = article_body.select(
        "pre, .code-block, .code-container, .highlight, .cm-content, div[class*='code'], div[class*='Code']"
    )
    for code_wrap in code_candidates:
        if code_wrap.parent is None:
            continue
        # Skip small inline wrappers; only process multi-line or block code containers
        if code_wrap.name == "div" and not code_wrap.find(["pre", "code", "table", "br"]) and len(code_wrap.get_text(strip=True).splitlines()) < 2:
            continue

        for noise in code_wrap.select(".gutter, .line-numbers, .linenumber, .line-number, td.gutter, .code-header, .toolbar, .copy-btn, button"):
            if noise.parent is not None:
                noise.decompose()

        for br in code_wrap.find_all("br"):
            br.replace_with("\n")

        # Only add newlines to block-level line elements (not inline token divs/spans)
        for line_el in code_wrap.select("tr, .cm-line, .code-line, p"):
            line_el.append("\n")

        lines = [line.rstrip() for line in code_wrap.get_text().splitlines()]
        # Collapse excessive blank lines
        clean_code = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if clean_code:
            new_pre = soup.new_tag("pre")
            new_code = soup.new_tag("code")
            new_code.string = clean_code
            new_pre.append(new_code)
            code_wrap.replace_with(new_pre)

    # 4. Fix image URLs, deduplicate repeated images, and remove sibling OCR text dumps
    images = []
    seen_srcs = set()
    for img in list(article_body.find_all("img")):
        src = img.get("src") or img.get("data-src")
        if not src and img.get("srcset"):
            src = img.get("srcset").split(",")[0].strip().split(" ")[0]

        if not src or src.startswith("data:image/svg"):
            img.decompose()
            continue

        if src.startswith("//"):
            src = "https:" + src

        norm_src = src.split("?")[0]
        if norm_src in seen_srcs:
            img.decompose()
            continue
        is_tiny = False
        for dim_attr in ("width", "height"):
            dim_val = img.get(dim_attr)
            if dim_val and str(dim_val).strip().isdigit() and int(str(dim_val).strip()) < 40:
                is_tiny = True
                break
        if is_tiny:
            img.decompose()
            continue
        seen_srcs.add(norm_src)
        img["src"] = src
        if img.has_attr("style"):
            del img["style"]
        images.append({"src": src, "alt": img.get("alt", "")})

    title_tag = article_body.find("h1") or soup.find("h1") or soup.title
    title = title_tag.get_text(" ", strip=True) if title_tag else "Study Notes"

    clean_html = render_tex_to_html(str(article_body))

    return {
        "title": title,
        "source_url": url,
        "clean_html": clean_html,
        "image_count": len(images)
    }

if __name__ == "__main__":
    test_url = "https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/"
    data = asyncio.run(extract_clean_article(test_url))
    print(f"Title: {data['title']}")
    print(f"Unique images extracted: {data['image_count']}")

    with open("extracted_raw.html", "w", encoding="utf-8") as f:
        f.write(data["clean_html"])
