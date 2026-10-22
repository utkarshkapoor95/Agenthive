import os
import re
import copy
import html
import time
import uuid
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from graph import graph, AGENT_ORDER, DEPENDENCIES
from agents import run_intake_agent, run_website_edit, fetch_site_images
from tools import (
    BufferError,
    buffer_get_channel,
    buffer_publish_now,
    buffer_get_post,
    buffer_delete_post,
)

load_dotenv()

app = FastAPI(title="AgentHive")

BASE_DIR = Path(__file__).parent
SITES_DIR = BASE_DIR / "generated_sites"
SITES_DIR.mkdir(exist_ok=True)
TEMPLATE_PATH = BASE_DIR / "website_template.html"
FRONTEND_PATH = BASE_DIR / "index.html"

# ---------------------------------------------------------------
# Demo publishing config (all from .env)
#   BUFFER_API_KEY        key for the Buffer account that owns the demo channel
#   DEMO_CHANNEL_ID       Buffer channel id of the demo Instagram account
#   DEMO_CHANNEL_NAME     expected channel name (blocks posting if it does not match)
#   DEMO_PROFILE_URL      public link visitors can open, e.g. https://instagram.com/yourdemo
#   DEMO_DAILY_CAP        max live demo posts per day across all visitors (default 3)
#   DEMO_PUBLISH_ENABLED  set to "false" to switch live posting off
# ---------------------------------------------------------------
DEMO_CHANNEL_ID = os.environ.get("DEMO_CHANNEL_ID", "").strip()
DEMO_CHANNEL_NAME = os.environ.get("DEMO_CHANNEL_NAME", "").strip().lstrip("@").lower()
DEMO_PROFILE_URL = os.environ.get("DEMO_PROFILE_URL", "").strip()
DEMO_DAILY_CAP = int(os.environ.get("DEMO_DAILY_CAP", "3") or 3)
DEMO_PUBLISH_ENABLED = os.environ.get("DEMO_PUBLISH_ENABLED", "true").strip().lower() != "false"

PUBLISH_COOLDOWN_SECONDS = 60       # per visitor IP
STATUS_MIN_INTERVAL_SECONDS = 8     # protects Buffer's 250 requests per day
LAUNCHES_PER_IP_PER_HOUR = 6
MAX_RUNS_IN_MEMORY = 100


def demo_ready() -> bool:
    return bool(os.environ.get("BUFFER_API_KEY") and DEMO_CHANNEL_ID and DEMO_PUBLISH_ENABLED)


# ---------------------------------------------------------------
# In-memory state (resets when the server restarts)
# ---------------------------------------------------------------
RUNS: "OrderedDict[str, dict]" = OrderedDict()
LOCK = threading.RLock()
RUN_SLOT = threading.Semaphore(1)   # one pipeline at a time (Groq: 8K tokens per minute)
PUBLISHES: dict = {}                # post_id -> {"run_id", "final", "last_check", "result"}
DAILY = {"date": "", "count": 0}
LAST_PUBLISH_BY_IP: dict = {}
LAUNCHES_BY_IP: dict = {}
_CHANNEL_OK = {"checked": False}


class IdeaRequest(BaseModel):
    idea: str


class LaunchRequest(BaseModel):
    idea: str
    context: dict = {}


class PublishRequest(BaseModel):
    run_id: str
    post_index: int = 0


class EditWebsiteRequest(BaseModel):
    run_id: str
    edit_request: str


# ---------------------------------------------------------------
# Small safety helpers
# ---------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def safe_hex(color, default: str) -> str:
    color = str(color or "").strip()
    return color if re.fullmatch(r"#[0-9a-fA-F]{6}", color) else default


def safe_img(url) -> str:
    url = str(url or "").strip()
    return url if url.startswith("https://images.pexels.com/") else ""


def as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def make_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower()) or "yourstartup"


BLOCKED_WORDS = [
    "fuck", "shit", "bitch", "asshole", "porn", "nude", "nudes", "sex", "sexy", "rape",
    "kill", "suicide", "nazi", "terror", "terrorist", "bomb", "weapon", "gun", "cocaine",
    "heroin", "escort", "casino", "betting",
]
_BLOCK_RE = re.compile(r"\b(" + "|".join(BLOCKED_WORDS) + r")\b", re.IGNORECASE)


def moderate_text(*texts) -> bool:
    """Basic keyword screen for the public demo. Returns True when the text is OK."""
    return not any(_BLOCK_RE.search(t or "") for t in texts)


def rate_limit(store: dict, key: str, limit: int, window: float) -> bool:
    now = time.time()
    with LOCK:
        hits = [t for t in store.get(key, []) if now - t < window]
        if len(hits) >= limit:
            store[key] = hits
            return False
        hits.append(now)
        store[key] = hits
        return True


# ---------------------------------------------------------------
# Website generation (same template and placeholders as before)
# ---------------------------------------------------------------
def get_img(images, index) -> str:
    images = images or []
    if index < len(images):
        return safe_img(as_dict(images[index]).get("src_large") or as_dict(images[index]).get("src_medium"))
    if images:
        return safe_img(as_dict(images[0]).get("src_large") or as_dict(images[0]).get("src_medium"))
    return ""


def generate_website_html(website_content: dict, images: list) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    wc = as_dict(website_content)

    colors = as_dict(wc.get("colors"))
    primary = safe_hex(colors.get("primary"), "#6366f1")
    secondary = safe_hex(colors.get("secondary"), "#1e1b4b")
    accent = safe_hex(colors.get("accent"), "#f59e0b")

    company_name = str(wc.get("company_name") or "Your Startup")
    slug = make_slug(company_name)

    how_html = ""
    for i, item in enumerate(wc.get("how_it_works") or []):
        item = as_dict(item)
        how_html += (
            f'<div class="step-card card-3d reveal reveal-delay-{min(i, 3)}">'
            f'<div class="step-number">{esc(item.get("step", i + 1))}</div>'
            f'<h3>{esc(item.get("title"))}</h3><p>{esc(item.get("description"))}</p></div>\n'
        )

    features_html = ""
    for i, f in enumerate(wc.get("features") or []):
        f = as_dict(f)
        features_html += (
            f'<div class="feature-card card-3d reveal reveal-delay-{min(i, 3)}">'
            f'<h3>{esc(f.get("title"))}</h3><p>{esc(f.get("description"))}</p></div>\n'
        )

    testimonials_html = ""
    for i, t in enumerate(wc.get("testimonials") or []):
        t = as_dict(t)
        name = str(t.get("name") or "Customer")
        testimonials_html += (
            f'<div class="testimonial-card card-3d reveal reveal-delay-{min(i, 3)}">'
            f'<p class="quote">"{esc(t.get("quote"))}"</p>'
            f'<div class="testimonial-author"><div class="author-avatar">{esc(name[:1].upper())}</div>'
            f'<div><div class="author-name">{esc(name)}</div>'
            f'<div class="author-role">{esc(t.get("role"))}</div></div></div></div>\n'
        )

    pricing_html = ""
    for p in wc.get("pricing") or []:
        p = as_dict(p)
        highlighted = bool(p.get("highlighted"))
        card_class = "price-card card-3d highlighted" if highlighted else "price-card card-3d"
        badge = '<div class="popular-badge">Most Popular</div>' if highlighted else ""
        feats = p.get("features") or []
        features_list = "".join(f"<li>{esc(feat)}</li>" for feat in feats if isinstance(feat, (str, int, float)))
        btn_class = "price-cta-fill" if highlighted else "price-cta-outline"
        pricing_html += (
            f'<div class="{card_class}">{badge}'
            f'<div class="plan-name">{esc(p.get("plan"))}</div>'
            f'<div class="price">{esc(p.get("price"))}</div>'
            f'<ul>{features_list}</ul>'
            f'<a href="#contact" class="{btn_class}">Get Started</a></div>\n'
        )

    faq_html = ""
    for fq in wc.get("faq") or []:
        fq = as_dict(fq)
        faq_html += (
            f'<details class="faq-item reveal"><summary>{esc(fq.get("question"))}</summary>'
            f'<p>{esc(fq.get("answer"))}</p></details>\n'
        )

    values = {
        "{{COMPANY_NAME}}": esc(company_name),
        "{{TAGLINE}}": esc(wc.get("tagline")),
        "{{HERO_DESCRIPTION}}": esc(wc.get("hero_description")),
        "{{CTA_TEXT}}": esc(wc.get("cta_text") or "Get Started"),
        "{{HERO_IMAGE}}": get_img(images, 0),
        "{{IMAGE_2}}": get_img(images, 1),
        "{{IMAGE_3}}": get_img(images, 2),
        "{{IMAGE_4}}": get_img(images, 3),
        "{{PRIMARY_COLOR}}": primary,
        "{{SECONDARY_COLOR}}": secondary,
        "{{ACCENT_COLOR}}": accent,
        "{{ABOUT_TEXT}}": esc(wc.get("about") or "We are building something great."),
        "{{HOW_IT_WORKS_HTML}}": how_html,
        "{{FEATURES_HTML}}": features_html,
        "{{TESTIMONIALS_HTML}}": testimonials_html,
        "{{PRICING_HTML}}": pricing_html,
        "{{FAQ_HTML}}": faq_html,
        "{{CONTACT_PHONE}}": "+91 98765 43210",
        "{{CONTACT_EMAIL}}": f"info@{slug}.com",
        "{{CONTACT_WHATSAPP}}": f"https://wa.me/919876543210?text=Hi%20{slug}",
        "{{SOCIAL_INSTAGRAM}}": f"https://instagram.com/{slug}",
        "{{SOCIAL_TWITTER}}": f"https://twitter.com/{slug}",
        "{{SOCIAL_FACEBOOK}}": f"https://facebook.com/{slug}",
    }
    out = template
    for key, val in values.items():
        out = out.replace(key, val)
    return out


# ---------------------------------------------------------------
# Pipeline runner (background thread, live progress)
# ---------------------------------------------------------------
def _new_run(idea: str, context: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    run = {
        "run_id": run_id,
        "idea": idea,
        "context": context,
        "queued": True,
        "done": False,
        "error": None,
        "agents": {name: "pending" for name in AGENT_ORDER},
        "market": None,
        "website": None,
        "customers": None,
        "content": None,
        "publish": None,
        "created_at": time.time(),
    }
    with LOCK:
        RUNS[run_id] = run
        while len(RUNS) > MAX_RUNS_IN_MEMORY:
            RUNS.popitem(last=False)
    return run_id


def _set_running_if_ready(run: dict):
    for name in AGENT_ORDER:
        if run["agents"][name] == "pending" and all(run["agents"][d] == "done" for d in DEPENDENCIES[name]):
            run["agents"][name] = "running"


def _on_node_done(run: dict, node: str, out: dict):
    with LOCK:
        if node == "market":
            run["market"] = out.get("market_summary")
        elif node == "website":
            wc = as_dict(out.get("website_content"))
            images = out.get("site_images") or []
            site_id = uuid.uuid4().hex[:8]
            (SITES_DIR / f"{site_id}.html").write_text(generate_website_html(wc, images), encoding="utf-8")
            colors = as_dict(wc.get("colors"))
            run["_website_content"] = wc  # full dict for the edit endpoint
            run["website"] = {
                "company_name": wc.get("company_name"),
                "tagline": wc.get("tagline"),
                "hero_description": wc.get("hero_description"),
                "preview_url": f"/preview/{site_id}",
                "hero_image": get_img(images, 0),
                "colors": {
                    "primary": safe_hex(colors.get("primary"), "#6366f1"),
                    "secondary": safe_hex(colors.get("secondary"), "#1e1b4b"),
                    "accent": safe_hex(colors.get("accent"), "#f59e0b"),
                },
            }
        elif node == "customers":
            run["customers"] = out.get("customers")
        elif node == "content":
            run["content"] = out.get("content")
        run["agents"][node] = "done"
        _set_running_if_ready(run)


def _run_pipeline(run_id: str, idea: str, context: dict):
    run = RUNS.get(run_id)
    if run is None:
        return
    with RUN_SLOT:  # waits here while another launch is running
        with LOCK:
            run["queued"] = False
            run["started_at"] = time.time()
            _set_running_if_ready(run)
        try:
            initial = {
                "idea": idea,
                "founder_context": context or None,
                "market_summary": None,
                "website_content": None,
                "site_images": None,
                "customers": None,
                "content": None,
                "status": "starting",
            }
            for chunk in graph.stream(initial, stream_mode="updates"):
                for node, out in chunk.items():
                    if node in run["agents"]:
                        _on_node_done(run, node, out or {})
            with LOCK:
                run["done"] = True
        except Exception as e:  # keep the server alive, show the error in the UI
            print(f"[PIPELINE] ERROR: {e}")
            with LOCK:
                run["error"] = str(e)[:300]
                for name, state in run["agents"].items():
                    if state == "running":
                        run["agents"][name] = "error"
                run["done"] = True


def _snapshot(run: dict) -> dict:
    with LOCK:
        snap = {k: copy.deepcopy(run[k]) for k in (
            "run_id", "idea", "queued", "done", "error", "agents",
            "market", "website", "customers", "content", "publish",
        )}
        snap["elapsed"] = round(time.time() - run.get("started_at", time.time()), 1) if not run["queued"] else 0
    return snap


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if FRONTEND_PATH.exists():
        return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AgentHive is running.</h1>")


@app.get("/health")
def health_check():
    return {"status": "running"}


@app.get("/config")
def config():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with LOCK:
        used = DAILY["count"] if DAILY["date"] == today else 0
    return {
        "demo_publish": demo_ready(),
        "profile_url": DEMO_PROFILE_URL,
        "daily_cap": DEMO_DAILY_CAP,
        "remaining_today": max(DEMO_DAILY_CAP - used, 0),
    }


@app.post("/intake")
def intake(request: IdeaRequest, http: Request):
    idea = request.idea.strip()
    if not 3 <= len(idea) <= 300:
        raise HTTPException(400, "Describe your idea in 3 to 300 characters.")
    if not moderate_text(idea):
        raise HTTPException(400, "This idea cannot be used in the public demo. Try a different business idea.")
    return run_intake_agent(idea)


@app.post("/launch")
def launch(request: LaunchRequest, http: Request):
    idea = request.idea.strip()
    if not 3 <= len(idea) <= 300:
        raise HTTPException(400, "Describe your idea in 3 to 300 characters.")
    context = {str(k)[:40]: str(v)[:200] for k, v in (request.context or {}).items() if v}
    if not moderate_text(idea, *context.values()):
        raise HTTPException(400, "This idea cannot be used in the public demo. Try a different business idea.")
    if not rate_limit(LAUNCHES_BY_IP, client_ip(http), LAUNCHES_PER_IP_PER_HOUR, 3600):
        raise HTTPException(429, "Too many launches from this connection. Try again in an hour.")

    run_id = _new_run(idea, context)
    threading.Thread(target=_run_pipeline, args=(run_id, idea, context), daemon=True).start()
    return {"run_id": run_id}


@app.get("/run/{run_id}")
def get_run(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "Run not found. The server may have restarted. Start a new launch.")
    return _snapshot(run)


@app.post("/edit-website")
def edit_website(request: EditWebsiteRequest, http: Request):
    run = RUNS.get(request.run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    if not run.get("done"):
        raise HTTPException(400, "The run is still in progress. Wait for it to finish.")
    edit_text = request.edit_request.strip()
    if not 3 <= len(edit_text) <= 500:
        raise HTTPException(400, "Describe your changes in 3 to 500 characters.")
    if not moderate_text(edit_text):
        raise HTTPException(400, "This edit request cannot be processed in the public demo.")

    # Get the current website content from the run's stored data
    # We need the original website_content dict, not the trimmed version stored in run["website"]
    # So we re-read it from the graph state or re-generate from the edit
    current_wc = run.get("_website_content")
    if not current_wc:
        raise HTTPException(400, "No website data available to edit.")

    idea = run["idea"]
    context = run.get("context") or {}

    try:
        new_wc = run_website_edit(current_wc, edit_text, idea, context)
        images = fetch_site_images(new_wc, idea)
        site_id = uuid.uuid4().hex[:8]
        (SITES_DIR / f"{site_id}.html").write_text(generate_website_html(new_wc, images), encoding="utf-8")
        colors = as_dict(new_wc.get("colors"))
        with LOCK:
            run["_website_content"] = new_wc
            run["website"] = {
                "company_name": new_wc.get("company_name"),
                "tagline": new_wc.get("tagline"),
                "hero_description": new_wc.get("hero_description"),
                "preview_url": f"/preview/{site_id}",
                "hero_image": get_img(images, 0),
                "colors": {
                    "primary": safe_hex(colors.get("primary"), "#6366f1"),
                    "secondary": safe_hex(colors.get("secondary"), "#1e1b4b"),
                    "accent": safe_hex(colors.get("accent"), "#f59e0b"),
                },
            }
        return {"ok": True, "website": run["website"]}
    except Exception as e:
        print(f"[EDIT WEBSITE] ERROR: {e}")
        raise HTTPException(500, f"Could not apply the edit: {str(e)[:200]}")


def _verify_demo_channel():
    """Refuse to post unless the configured channel is the expected Instagram demo channel."""
    if _CHANNEL_OK["checked"]:
        return
    ch = buffer_get_channel(DEMO_CHANNEL_ID)
    if not ch:
        raise HTTPException(503, "Demo channel not found in Buffer. Live posting is blocked.")
    if str(ch.get("service", "")).lower() != "instagram":
        raise HTTPException(503, "Demo channel is not an Instagram channel. Live posting is blocked.")
    if DEMO_CHANNEL_NAME:
        names = {str(ch.get("name", "")).lstrip("@").lower(), str(ch.get("displayName", "")).lstrip("@").lower()}
        if DEMO_CHANNEL_NAME not in names:
            raise HTTPException(503, "Demo channel name does not match DEMO_CHANNEL_NAME. Live posting is blocked.")
    print(f"[PUBLISH] Demo channel verified: {ch.get('name')} ({ch.get('service')})")
    _CHANNEL_OK["checked"] = True


@app.post("/publish-demo")
def publish_demo(request: PublishRequest, http: Request):
    if not demo_ready():
        raise HTTPException(503, "Live posting is switched off on this server. Copy the post instead.")

    run = RUNS.get(request.run_id)
    if run is None or not run.get("content"):
        raise HTTPException(404, "No generated posts found for this run.")
    if run.get("publish"):
        raise HTTPException(409, "This run already posted to the demo account.")

    posts = as_dict(run["content"]).get("posts") or []
    if not posts:
        raise HTTPException(400, "This run has no posts to publish.")
    idx = max(0, min(request.post_index, len(posts) - 1))
    post = as_dict(posts[idx])  # the browser picks which post, but cannot send its own text

    caption = str(post.get("caption", "")).strip()
    tags = " ".join(str(t) for t in (post.get("hashtags") or [])[:8])
    image_url = safe_img(post.get("image_url"))
    if not caption or not image_url:
        raise HTTPException(400, "The first post has no caption or image, so it cannot be published.")
    if not moderate_text(run["idea"], caption, tags):
        raise HTTPException(400, "This post did not pass the demo content check and was not published.")

    ip = client_ip(http)
    if not rate_limit(LAST_PUBLISH_BY_IP, ip, 1, PUBLISH_COOLDOWN_SECONDS):
        raise HTTPException(429, "Please wait a minute before posting again.")

    # Reserve a slot in today's cap before calling Buffer.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with LOCK:
        if DAILY["date"] != today:
            DAILY["date"], DAILY["count"] = today, 0
        if DAILY["count"] >= DEMO_DAILY_CAP:
            raise HTTPException(429, "Today's live demo limit is reached. The post was not published. Copy it instead.")
        DAILY["count"] += 1

    try:
        _verify_demo_channel()
        text = (
            f"{caption}\n\n{tags}\n\n#AgentHiveDemo\n"
            "Generated by AgentHive, a multi-agent AI startup launch kit (demo post)."
        ).strip()
        created = buffer_publish_now(DEMO_CHANNEL_ID, text[:2000], image_url)
    except HTTPException:
        with LOCK:
            DAILY["count"] = max(DAILY["count"] - 1, 0)
        raise
    except BufferError as e:
        with LOCK:
            DAILY["count"] = max(DAILY["count"] - 1, 0)
        print(f"[PUBLISH] Buffer error: {e}")
        raise HTTPException(502, f"This post could not be published: {e}")

    post_id = created.get("id")
    with LOCK:
        PUBLISHES[post_id] = {"run_id": request.run_id, "final": None, "last_check": 0.0, "result": None,
                              "started": time.time()}
        run["publish"] = {"post_id": post_id, "state": "pending"}
    return {"post_id": post_id, "state": "pending", "profile_url": DEMO_PROFILE_URL}


@app.get("/publish-status/{post_id}")
def publish_status(post_id: str):
    with LOCK:
        info = PUBLISHES.get(post_id)
    if info is None:
        raise HTTPException(404, "Unknown post.")
    if info["final"]:
        return info["result"]

    now = time.time()
    if now - info["last_check"] < STATUS_MIN_INTERVAL_SECONDS and info["result"]:
        return info["result"]

    try:
        p = buffer_get_post(post_id)
    except BufferError as e:
        return info["result"] or {"state": "pending", "message": str(e)}

    status = p.get("status", "")
    elapsed = round(now - info["started"], 1)
    if status == "sent":
        result = {"state": "live", "link": p.get("externalLink") or DEMO_PROFILE_URL,
                  "profile_url": DEMO_PROFILE_URL, "seconds": elapsed}
        final = True
    elif status == "error":
        result = {"state": "failed", "message": p.get("error") or "Instagram rejected the post.",
                  "profile_url": DEMO_PROFILE_URL, "seconds": elapsed}
        final = True
        try:  # keep the demo queue clean
            buffer_delete_post(post_id)
        except BufferError:
            pass
    else:
        result = {"state": "pending", "buffer_status": status, "seconds": elapsed,
                  "profile_url": DEMO_PROFILE_URL}
        final = False

    with LOCK:
        info["last_check"], info["result"], info["final"] = now, result, final
        run = RUNS.get(info["run_id"])
        if run is not None and run.get("publish"):
            run["publish"]["state"] = result["state"]
            if result.get("link"):
                run["publish"]["link"] = result["link"]
    return result


@app.get("/preview/{site_id}", response_class=HTMLResponse)
def preview_site(site_id: str):
    if not re.fullmatch(r"[a-f0-9]{8}", site_id):
        return HTMLResponse("<h1>Site not found</h1>", status_code=404)
    site_path = SITES_DIR / f"{site_id}.html"
    if site_path.exists():
        return HTMLResponse(site_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Site not found</h1>", status_code=404)
