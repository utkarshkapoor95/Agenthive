# AgentHive

**Multi-Agent AI Startup Launch Kit**

Describe your business in one sentence. Five AI agents research your city, build your website, find your first customers, write your first week of posts, and plan your first 30 days. One of those posts goes live on a real Instagram account.

**[Try it live](https://agenthive.autothynk.com)**

---

## What it does

A founder types a one-sentence business idea ("A bakery in Jaipur that delivers fresh bread daily"). AgentHive asks 6 tailored follow-up questions, then runs 5 specialized agents that produce:

- **Market research** with 5 local competitors, their websites, gap analysis, and a target customer profile
- **A generated website** with Pexels photos, pricing, FAQ, testimonials, and an edit feature where the founder describes changes and the agent rebuilds the site
- **Potential customers and partners** across 3 segments, with phone numbers, emails, and websites found via web search
- **5 Instagram posts** with unique images, a Google Business launch post, a WhatsApp announcement, and a content calendar. Any post can be published live to a real Instagram account
- **A 30-day launch plan** specific to the founder's city, budget, and business type

The full pipeline runs in under 5 minutes.

---

## Architecture

```
                    +------------------+
                    |   Intake Agent   |
                    | 6 follow-up Qs   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Market Research   |
                    | Agent             |
                    | 5 competitors,    |
                    | gap analysis      |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +------------v---------+
    | Website Builder    |     | Customer Finder      |
    | Agent              |     | Agent                |
    |                    |     |                      |
    | Landing page with  |     | 3 segments, 2-3      |
    | Pexels photos,     |     | targets each, with   |
    | edit capability    |     | contact enrichment   |
    +---------+----------+     +------------+----------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v---------+
                    | Social Media     |
                    | Agent            |
                    |                  |
                    | 5 Instagram      |
                    | posts + images,  |
                    | GMB post,        |
                    | WhatsApp msg,    |
                    | 30-day plan      |
                    +------------------+
```

**Flow:** Intake collects context. Market Research runs first (its output feeds all downstream agents). Website Builder and Customer Finder run in parallel. Social Media runs last, using market data to write relevant content.

**Orchestration:** LangGraph manages the agent graph with `Annotated[str, take_last]` reducers and `time.sleep(3)` pacing between nodes to respect Groq's rate limits.

---

## Tech stack

| Layer | Tool | Role |
|-------|------|------|
| LLM | [Groq](https://groq.com) (gpt-oss-120b) | All agent reasoning and content generation |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Agent graph with parallel execution and state management |
| Web search | [Tavily](https://tavily.com) | Competitor research, contact enrichment, market data |
| Images | [Pexels](https://pexels.com) | Website photos and Instagram post images |
| Social publishing | [Buffer API](https://buffer.com) | Live Instagram posting from the dashboard |
| Backend | FastAPI + Uvicorn | REST API with background thread pipeline and live progress |
| Frontend | Single HTML file | Dashboard with sidebar, collapsible cards, post previews |
| Hosting | DigitalOcean (Ubuntu 24.04) | Nginx reverse proxy, Let's Encrypt HTTPS |

---

## Demo Instagram

Posts published through AgentHive appear on [@agenthive.demo](https://instagram.com/agenthive.demo).

Publishing guardrails: daily cap (15 posts), 60-second cooldown per IP, keyword filter, channel lock. The server picks the post text from stored data. The browser cannot inject custom text.

---

## Key design decisions

**Why Groq over OpenAI?** Zero cost. Groq's free tier (30 RPM, 8K TPM) is enough for a demo. The retry logic in `tools.py` handles 429s with 15s/30s/45s backoff.

**Why Buffer over direct Instagram API?** Instagram's API requires app review and business verification. Buffer handles OAuth for all social platforms. One GraphQL call publishes to any connected channel.

**Why a single HTML frontend?** No build step, no React, no npm. The 45KB `index.html` includes the landing hero, intake form, agent dashboard with live status, Instagram-style post previews, website edit feature, and 30-day timeline. It loads in under 1 second.

**Why parallel execution?** Website Builder and Customer Finder don't depend on each other, only on Market Research output. Running them in parallel cuts total pipeline time from ~4.5 minutes to ~3.5 minutes.

---

## Run locally

**Prerequisites:** Python 3.12, API keys for Groq, Tavily, and Pexels (all free).

```bash
git clone https://github.com/utkarshkapoor95/Agenthive.git
cd Agenthive

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
PEXELS_API_KEY=your_key
```

Run:

```bash
uvicorn server:app --port 8000
```

Open `http://127.0.0.1:8000`. Do not use `--reload` if the folder is synced with OneDrive (causes restart loops).

---

## Project structure

```
agents.py             5 agents: Intake, Market, Website, Customer Finder, Social Media
graph.py              LangGraph flow with parallel join and rate-limit pacing
server.py             FastAPI endpoints, background pipeline, demo publishing guards
tools.py              Groq wrapper (retry + max_tokens), Tavily, Pexels, Buffer GraphQL
index.html            Full dashboard frontend
website_template.html Generated landing page template
```

---

## Limitations

This is a portfolio project, not a production product.

- **Groq free tier:** 8K tokens per minute. Only one user can run the pipeline at a time. Concurrent runs will hit rate limits.
- **Tavily:** 1,000 free searches per month. Each run uses 7-10 searches.
- **No persistent storage:** Run history lives in memory and resets when the server restarts.
- **Generated websites** are templates. They look good but are not production-ready.

---

## Built by

**[Utkarsh Kapoor](https://github.com/utkarshkapoor95)** | MBA @ ABV-IIITM Gwalior | Founder, [Autothynk](https://getautothynk.com)

AgentHive demonstrates multi-agent orchestration applied to a real GTM use case: taking a founder from idea to launch-ready in under 5 minutes.
