import re
import time
from urllib.parse import urlparse, quote_plus

from tools import call_groq, parse_json_response, search_web, search_images

# Set to False to skip the 30-day plan call inside the Content agent
# (saves one Groq call per run).
LAUNCH_PLAN_ENABLED = True


# =============================================================
# SHARED HELPERS
# =============================================================

def build_context_string(idea: str, context: dict) -> str:
    lines = [f"Business idea: {idea}"]
    for key, value in (context or {}).items():
        if value:
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def extract_short_idea(idea: str) -> str:
    """Clean up the idea into a short search-friendly phrase."""
    lower = idea.lower()
    for prefix in ["i want to start my ", "i want to start a ", "i want to open a ",
                   "i want to build a ", "i want to create a ", "i want to launch a ",
                   "start a ", "open a ", "build a ", "create a ", "launch a "]:
        if lower.startswith(prefix):
            lower = lower[len(prefix):]
            break
    for suffix in [" business", " startup", " company", " venture"]:
        if lower.endswith(suffix):
            lower = lower[:len(lower) - len(suffix)]
    return lower.strip()


def market_brief(market_summary: dict) -> str:
    """Short text version of the market research, to keep prompts small."""
    market_summary = market_summary or {}
    gaps = market_summary.get("gap_analysis", [])
    if isinstance(gaps, list):
        gaps = "; ".join(str(g) for g in gaps[:2])
    target = market_summary.get("target_customer", "")
    if isinstance(target, dict):
        target = " | ".join(
            str(target.get(k, "")) for k in ("demographics", "pain_points", "where_to_find") if target.get(k)
        )
    return f"Market gaps: {gaps}\nTarget customer: {target}"


# =============================================================
# AGENT 0: INTAKE AGENT
# =============================================================

INTAKE_AGENT_SYSTEM = """You are a sharp startup advisor doing a quick intake interview.
Given a one-sentence business idea, generate exactly 6 follow-up questions
that will help you understand the founder's situation.

You MUST always include these questions (tailor the wording to the idea):
1. A question about what name they want for their startup/brand (id: "startup_name")
2. A question about their target city or region (id: "city")
3. A question about business model - online, offline, or both (id: "model")
4. A question about their experience in this field (id: "experience")
5. A question about starting budget (id: "budget")
6. A question about their ideal first customer (id: "customer")

Tailor each question to the specific idea.

You MUST respond with ONLY valid JSON. No markdown fences, no extra text:
{
  "questions": [
    {
      "id": "startup_name",
      "question": "What name do you want for your startup? Leave blank if you want us to suggest one.",
      "placeholder": "e.g. FreshBake, UrbanLoaf, or leave empty",
      "type": "text"
    },
    {
      "id": "city",
      "question": "Which city or area are you targeting?",
      "placeholder": "e.g. Jaipur, Mumbai, Delhi NCR",
      "type": "text"
    }
  ]
}

type can be "text" or "select". For "select", include an "options" array."""

INTAKE_FALLBACK = {
    "questions": [
        {"id": "startup_name", "question": "What name do you want for your startup?", "placeholder": "Leave blank for a suggestion", "type": "text"},
        {"id": "city", "question": "Which city are you targeting?", "placeholder": "e.g. Mumbai, Delhi", "type": "text"},
        {"id": "model", "question": "Online, offline, or both?", "placeholder": "", "type": "select", "options": ["Online only", "Offline", "Both"]},
        {"id": "experience", "question": "Your experience in this field?", "placeholder": "e.g. 2 years", "type": "text"},
        {"id": "budget", "question": "Starting budget range?", "placeholder": "e.g. 5-10 lakh", "type": "text"},
        {"id": "customer", "question": "Your ideal first customer?", "placeholder": "e.g. college students", "type": "text"},
    ]
}


def run_intake_agent(idea: str) -> dict:
    print("\n[INTAKE AGENT] Starting...")
    try:
        raw = call_groq(INTAKE_AGENT_SYSTEM, f"Business idea: {idea}")
        result = parse_json_response(raw)
        if not result.get("questions"):
            raise ValueError("No questions returned")
        print(f"[INTAKE AGENT] Generated {len(result['questions'])} questions")
        return result
    except Exception as e:
        print(f"[INTAKE AGENT] ERROR: {e}")
        return INTAKE_FALLBACK


# =============================================================
# AGENT 1: MARKET RESEARCH AGENT
# =============================================================

MARKET_AGENT_SYSTEM = """You are a thorough market research analyst for early-stage startups.
Given a business idea AND the founder's context (city, budget, model, experience),
provide a detailed market analysis.

Your analysis must include:
1. FIVE real competitors in or relevant to their city/region. For each, give the name
   and their key advantage (what makes them strong).
2. A detailed gap analysis with 3-4 specific actionable points about what is missing
   in the market and what a new entrant can do differently.
3. A detailed target customer profile.

Use the founder's city to make this LOCAL and specific.

You MUST respond with ONLY valid JSON:
{
  "competitors": [
    {"name": "Competitor Name", "advantage": "Their key strength in one sentence"},
    {"name": "Competitor 2", "advantage": "Their key strength"},
    {"name": "Competitor 3", "advantage": "Their key strength"},
    {"name": "Competitor 4", "advantage": "Their key strength"},
    {"name": "Competitor 5", "advantage": "Their key strength"}
  ],
  "gap_analysis": [
    "First specific gap or opportunity",
    "Second specific gap",
    "Third specific gap",
    "Fourth specific gap"
  ],
  "target_customer": {
    "demographics": "Age, income, location details",
    "behavior": "How they currently solve this problem",
    "pain_points": "What frustrates them about current options",
    "where_to_find": "Where to reach them (channels, places, platforms)"
  }
}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _attach_competitor_websites(competitors: list, web_results: list, city: str = "") -> None:
    """
    Add a 'website' field to each competitor dict. First tries to match
    against Tavily results already fetched. If no match, falls back to a
    Google search link so the founder always has something to click.
    """
    for comp in competitors:
        if not isinstance(comp, dict):
            continue
        comp["website"] = ""
        name = comp.get("name", "")
        name_norm = _norm(name)
        if not name_norm:
            continue
        name_tokens = [t for t in name_norm.split() if len(t) > 2]
        matched = False
        if name_tokens:
            for r in web_results:
                title_norm = _norm(r.get("title", ""))
                content_norm = _norm(r.get("content", ""))
                full_match = name_norm in title_norm or name_norm in content_norm
                tokens_match = len(name_tokens) >= 2 and all(t in title_norm for t in name_tokens)
                if full_match or tokens_match:
                    comp["website"] = r.get("url", "")
                    matched = True
                    break
        if not matched:
            search_q = f"{name} {city}".strip() if city else name
            comp["website"] = f"https://www.google.com/search?q={quote_plus(search_q)}"


def run_market_agent(idea: str, context: dict = None) -> dict:
    print("\n[MARKET AGENT] Starting...")
    try:
        prompt = build_context_string(idea, context) if context else f"Business idea: {idea}"
        city = (context or {}).get("city", "")
        short_idea = extract_short_idea(idea)
        web_results = []

        if city:
            search_query = f"best {short_idea} in {city}"
            print(f"[MARKET AGENT] Searching: {search_query}")
            web_results = search_web(search_query, max_results=5)
            if web_results:
                search_text = "\n".join([f"- {r['title']}: {r['content'][:200]}" for r in web_results])
                prompt += f"\n\nWeb research results for {city}:\n{search_text}"

        raw = call_groq(MARKET_AGENT_SYSTEM, prompt)
        result = parse_json_response(raw)
        _attach_competitor_websites(result.get("competitors", []), web_results, city)
        print(f"[MARKET AGENT] Parsed keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"[MARKET AGENT] ERROR: {e}")
        return {
            "competitors": [{"name": "Unknown", "advantage": "Could not analyze"}],
            "gap_analysis": ["Could not analyze market"],
            "target_customer": {"demographics": "Unknown", "behavior": "Unknown", "pain_points": "Unknown", "where_to_find": "Unknown"},
        }


# =============================================================
# AGENT 2: WEBSITE AGENT
# =============================================================

WEBSITE_AGENT_SYSTEM = """You are a startup landing page copywriter.
Given a business idea, market research, and founder context, generate ALL content
for a professional, multi-section landing page.

RULES:
- If the founder provided a startup name, USE THAT NAME. Otherwise create one.
- Do NOT use emojis anywhere.
- Make copy specific to their city and business model.
- Use local currency (Rs or INR for India).
- Generate realistic testimonials with local names. Testimonials must be from REGULAR CUSTOMERS, not executives or founders. Use simple descriptions like "Regular customer", "Local resident", "First-time visitor", "Loyal customer since 2023". NEVER use titles like CEO, Founder, Director, Head of, VP, or any corporate designation. NEVER mention any company name. These should sound like real people leaving a Google review, not VIPs endorsing a brand.

You MUST respond with ONLY valid JSON:
{
  "company_name": "The startup name (use founder's choice if provided)",
  "tagline": "A compelling one-line tagline",
  "hero_description": "2-3 sentences about what the product does",
  "about": "2-3 sentences about the company story and mission",
  "how_it_works": [
    {"step": 1, "title": "Step title", "description": "One sentence"},
    {"step": 2, "title": "Step title", "description": "One sentence"},
    {"step": 3, "title": "Step title", "description": "One sentence"},
    {"step": 4, "title": "Step title", "description": "One sentence"}
  ],
  "features": [
    {"title": "Feature name", "description": "One sentence"},
    {"title": "Feature name", "description": "One sentence"},
    {"title": "Feature name", "description": "One sentence"},
    {"title": "Feature name", "description": "One sentence"}
  ],
  "testimonials": [
    {"name": "First Name", "role": "Regular customer", "quote": "1-2 sentences in casual, natural language"},
    {"name": "First Name", "role": "Local resident", "quote": "1-2 sentences in casual, natural language"},
    {"name": "First Name", "role": "Loyal customer", "quote": "1-2 sentences in casual, natural language"}
  ],
  "cta_text": "Call to action button text",
  "pricing": [
    {"plan": "Plan name", "price": "price", "features": ["f1", "f2", "f3"], "highlighted": false},
    {"plan": "Plan name", "price": "price", "features": ["f1", "f2", "f3", "f4"], "highlighted": true}
  ],
  "faq": [
    {"question": "Question?", "answer": "Answer."},
    {"question": "Question?", "answer": "Answer."},
    {"question": "Question?", "answer": "Answer."},
    {"question": "Question?", "answer": "Answer."}
  ],
  "colors": {
    "primary": "#hex",
    "secondary": "#hex (dark)",
    "accent": "#hex (CTA color)"
  },
  "image_keywords": ["keyword 1", "keyword 2", "keyword 3 (MUST describe the PHYSICAL business, NOT apps or technology. For a salon: salon haircut styling, beauty treatment spa. For a dentist: dental clinic patient, dentist treatment chair. For a bakery: fresh bread bakery counter, pastry display. Always about the real place and service, never phones, screens, or apps.)"]
}"""


def run_website_agent(idea: str, market_summary: dict, context: dict = None) -> dict:
    print("\n[WEBSITE AGENT] Starting...")
    try:
        market_summary = market_summary or {}
        if context:
            prompt = build_context_string(idea, context) + "\n\n"
            startup_name = context.get("startup_name", "")
            if startup_name:
                prompt += f"IMPORTANT: The founder wants the startup name to be: {startup_name}\n\n"
        else:
            prompt = f"Business idea: {idea}\n\n"

        comps = market_summary.get("competitors", [])
        comp_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in comps]
        prompt += f"Market research:\n- Competitors: {', '.join(comp_names[:3])}\n{market_brief(market_summary)}"

        raw = call_groq(WEBSITE_AGENT_SYSTEM, prompt)
        result = parse_json_response(raw)
        print(f"[WEBSITE AGENT] Parsed keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"[WEBSITE AGENT] ERROR: {e}")
        return {
            "company_name": (context or {}).get("startup_name") or "LaunchPad",
            "tagline": f"The smarter way to {idea.lower()}",
            "hero_description": f"We help you {idea.lower()}.",
            "about": "We are passionate about solving real problems.",
            "how_it_works": [
                {"step": 1, "title": "Sign Up", "description": "Create your account."},
                {"step": 2, "title": "Tell Us", "description": "Share your goals."},
                {"step": 3, "title": "We Build", "description": "Our team works."},
                {"step": 4, "title": "Results", "description": "Watch growth."},
            ],
            "features": [
                {"title": "Easy Setup", "description": "Get started in minutes."},
                {"title": "Smart Tools", "description": "AI-powered features."},
                {"title": "Always On", "description": "24/7 availability."},
                {"title": "No Lock-in", "description": "Cancel anytime."},
            ],
            "testimonials": [],
            "cta_text": "Get Started Free",
            "pricing": [
                {"plan": "Starter", "price": "Free", "features": ["Basic"], "highlighted": False},
                {"plan": "Pro", "price": "Rs 999/mo", "features": ["Everything"], "highlighted": True},
            ],
            "faq": [],
            "colors": {"primary": "#2563eb", "secondary": "#1e1b4b", "accent": "#10b981"},
            "image_keywords": [idea],
        }


def fetch_site_images(website_content: dict, idea: str) -> list:
    """
    Four Pexels photos for the generated website.
    Uses the LLM-generated image_keywords (which the prompt now constrains
    to physical business photos). Searches once with those keywords joined
    and gets 4 different photos in one Pexels call, the same approach that
    worked well in the earlier builds.
    """
    keywords = (website_content or {}).get("image_keywords") or []
    # Join the first 2-3 keywords into one search
    query = " ".join(str(k) for k in keywords[:3]).strip()
    if not query:
        query = extract_short_idea(idea)
    photos = search_images(query, per_page=4)
    if len(photos) < 2:
        # Fallback: try just the business type
        photos = search_images(extract_short_idea(idea), per_page=4)
    return photos[:4]


WEBSITE_EDIT_SYSTEM = """You are a startup landing page copywriter revising an existing website.
You are given the CURRENT website JSON and the founder's EDIT REQUEST.
Apply the requested changes to the website content. Keep everything else the same.
If the founder asks for different images, update the image_keywords array so Pexels
finds what they want.

You MUST respond with the COMPLETE updated website JSON (same structure as the original).
No markdown fences, no extra text, ONLY the full JSON object."""


def run_website_edit(current_website: dict, edit_request: str, idea: str, context: dict = None) -> dict:
    """Re-run the website agent with an edit request from the founder."""
    print(f"\n[WEBSITE EDIT] Request: {edit_request[:100]}")
    try:
        import json as _json
        current_json = _json.dumps(current_website, indent=2, ensure_ascii=False)
        prompt = (
            f"CURRENT WEBSITE:\n{current_json}\n\n"
            f"FOUNDER'S EDIT REQUEST:\n{edit_request}\n\n"
            f"Business idea: {idea}\n"
        )
        if context:
            prompt += build_context_string(idea, context) + "\n"
        raw = call_groq(WEBSITE_EDIT_SYSTEM, prompt, max_tokens=3500)
        result = parse_json_response(raw)
        print(f"[WEBSITE EDIT] Parsed keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"[WEBSITE EDIT] ERROR: {e}")
        return current_website


# =============================================================
# AGENT 3: CUSTOMER FINDER AGENT
# Finds organizations that could BUY from the founder or REFER
# customers. It never lists competitors.
# =============================================================

SEGMENTS_SYSTEM = """You help a brand-new local business find its first customers and referral partners.
Given the business idea, city and target customer, name exactly 3 groups of ORGANIZATIONS or
PLACES that could buy from this business or send it customers.
NEVER list competitors (businesses that offer the same service).
For each group, write a short search query a person would type into Google to find real
examples in that city.

You MUST respond with ONLY valid JSON:
{"segments":[{"segment":"short name, e.g. Corporate offices","why":"one sentence on why they would become customers or send customers","query":"search query that includes the city"}]}"""

CUSTOMER_EXTRACT_SYSTEM = """You are given customer segments and web search results for each segment.
For each segment, pick 2 or 3 REAL organizations that appear in that segment's search results
and write what the founder should offer them.

RULES:
- Use only names that appear in the search results. Never invent an organization.
- url must be copied from the search results for that organization. If unsure, use an empty string.
- phone and email: copy them ONLY if that exact phone number or email address appears in the
  text given to you for that organization. If none appears, leave them as empty strings. Never
  guess, construct, or reuse a phone/email from a different organization.
- Do not include competitors or the founder's own business.
- pitch: one concrete sentence (for example a bulk order, a referral arrangement, a free trial day).
- If a segment's results contain no real organization, return an empty targets list and give a
  one-sentence how_to_find tip.

You MUST respond with ONLY valid JSON:
{"segments":[{"segment":"...","why":"...","targets":[{"name":"...","url":"...","phone":"","email":"...","pitch":"..."}],"how_to_find":""}]}"""


def _clean_phone(raw: str) -> str:
    raw = str(raw or "").strip()
    digits = re.sub(r"\D", "", raw)
    return raw if 7 <= len(digits) <= 15 else ""


def _clean_email(raw: str) -> str:
    raw = str(raw or "").strip()
    return raw if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", raw) else ""


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def run_customer_agent(idea: str, market_summary: dict, context: dict = None) -> dict:
    print("\n[CUSTOMER FINDER] Starting...")
    city = (context or {}).get("city", "")
    try:
        short_idea = extract_short_idea(idea)
        seg_prompt = (
            f"Business: {short_idea}\nCity: {city or 'India'}\n"
            f"{market_brief(market_summary)}"
        )
        seg_raw = call_groq(SEGMENTS_SYSTEM, seg_prompt)
        segments = parse_json_response(seg_raw).get("segments", [])[:3]
        if not segments:
            raise ValueError("No segments returned")

        allowed_hosts = {}
        search_text = ""
        for i, seg in enumerate(segments, 1):
            query = seg.get("query") or f"{seg.get('segment', '')} in {city}"
            print(f"[CUSTOMER FINDER] Search {i}: {query}")
            results = search_web(query, max_results=4)
            hosts = set()
            search_text += f"\n## Segment {i}: {seg.get('segment', '')}\n"
            for r in results:
                hosts.add(_host(r["url"]))
                search_text += f"- {r['title']} | {r['url']} | {r['content'][:220]}\n"
            allowed_hosts[seg.get("segment", "")] = hosts
            time.sleep(1.5)

        raw = call_groq(
            CUSTOMER_EXTRACT_SYSTEM,
            f"Business: {short_idea}\nCity: {city or 'India'}\n{search_text}",
        )
        result = parse_json_response(raw)

        cleaned = []
        all_targets = []
        for seg in result.get("segments", [])[:3]:
            hosts = allowed_hosts.get(seg.get("segment", ""), set())
            targets = []
            for t in seg.get("targets", [])[:3]:
                name = str(t.get("name", "")).strip()
                if not name:
                    continue
                url = str(t.get("url", "")).strip()
                if not (url.startswith("http") and _host(url) in hosts):
                    url = f"https://www.google.com/maps/search/{quote_plus(name + ' ' + city)}"
                target = {
                    "name": name,
                    "url": url,
                    "phone": _clean_phone(t.get("phone", "")),
                    "email": _clean_email(t.get("email", "")),
                    "pitch": str(t.get("pitch", "")).strip(),
                }
                targets.append(target)
                all_targets.append(target)
            cleaned.append({
                "segment": seg.get("segment", ""),
                "why": seg.get("why", ""),
                "targets": targets,
                "how_to_find": seg.get("how_to_find", ""),
            })

        # Extra search per target to find phone/email when missing
        for target in all_targets:
            if target["phone"] and target["email"]:
                continue
            try:
                contact_query = f"{target['name']} {city} contact phone email"
                print(f"[CUSTOMER FINDER] Contact search: {contact_query}")
                contact_results = search_web(contact_query, max_results=3)
                all_text = " ".join(r.get("content", "") + " " + r.get("title", "") for r in contact_results)
                if not target["phone"]:
                    phones = re.findall(r"(?:\+91[\s-]?)?(?:\d[\d\s-]{7,12}\d)", all_text)
                    for p in phones:
                        cleaned_p = _clean_phone(p)
                        if cleaned_p:
                            target["phone"] = cleaned_p
                            break
                if not target["email"]:
                    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", all_text)
                    for e in emails:
                        cleaned_e = _clean_email(e)
                        if cleaned_e:
                            target["email"] = cleaned_e
                            break
                # Also try to upgrade the URL if it was a maps fallback
                if "google.com/maps/search" in target["url"] and contact_results:
                    for r in contact_results:
                        rurl = r.get("url", "")
                        if rurl.startswith("http") and "google.com" not in rurl:
                            target["url"] = rurl
                            break
                time.sleep(1)
            except Exception as e:
                print(f"[CUSTOMER FINDER] Contact search error for {target['name']}: {e}")

        print(f"[CUSTOMER FINDER] {sum(len(s['targets']) for s in cleaned)} targets in {len(cleaned)} segments")
        return {"segments": cleaned}
    except Exception as e:
        print(f"[CUSTOMER FINDER] ERROR: {e}")
        return {"segments": [], "error": str(e)}


# =============================================================
# AGENT 4: CONTENT AND LAUNCH AGENT
# One agent, two outputs: the first week of social content, and
# a 30-day launch plan that schedules it.
# =============================================================

CONTENT_SYSTEM = """You are the social media manager for a brand-new local business.
Write its first week of content.

RULES:
- No emojis anywhere.
- Each Instagram caption is 40 to 80 words, specific to this business, city and customer.
- No invented facts: no years in business, no statistics, no customer quotes, no awards.
- 6 hashtags per post, mixing the city, the niche and the offer. Each starts with #.
- image_keyword is 2 to 4 plain words describing a real photograph that matches this
  specific business and post topic. Be concrete: "fresh bread bakery counter" not "bakery",
  "dentist examining patient" not "dental". No text overlays, no logos, no illustrations.
- Posts are spread over the week with these day labels and types:
  Day 1 Launch announcement, Day 2 Behind the scenes, Day 4 Offer or tip,
  Day 5 Problem solved for a customer, Day 7 Weekend call to action.

You MUST respond with ONLY valid JSON:
{
  "posts": [
    {"day": "Day 1", "type": "Launch announcement", "caption": "...", "hashtags": ["#a", "#b", "#c", "#d", "#e", "#f"], "image_keyword": "..."}
  ],
  "gmb_post": "Google Business Profile launch post, 60 to 90 words, plain text",
  "whatsapp_announcement": "WhatsApp or Facebook launch message, 40 to 70 words, plain text"
}
The posts array must contain exactly 5 posts."""

PLAN_SYSTEM = """You are a launch advisor for a brand-new local business.
Write a 30-day launch plan with 4 weeks. Each week has a theme and exactly 3 concrete actions.

RULES:
- Be specific to the city, budget and business model given. Do not exceed the founder's budget.
- Use rupees (Rs) for any money amounts.
- Refer to the social posts and customer groups provided instead of repeating them.
- Each action is one sentence that starts with a verb.

You MUST respond with ONLY valid JSON:
{"weeks":[{"week":1,"theme":"short theme","actions":["...","...","..."]}]}
The weeks array must contain exactly 4 weeks."""


def _fetch_post_images(posts: list, idea: str) -> None:
    """
    Assign a unique Pexels image to each post. Searches per post with
    the business type prepended to the keyword so results stay on topic.
    Tracks used image URLs to prevent repeats across posts.
    """
    short = extract_short_idea(idea)
    used_urls = set()
    for p in posts:
        kw = str(p.get("image_keyword", "")).strip()
        # Prepend the business type so "preparation" becomes "salon preparation"
        query = f"{short} {kw}" if kw else short
        photos = search_images(query, per_page=3)
        chosen = None
        for photo in photos:
            url = photo.get("src_large", "")
            if url and url not in used_urls:
                chosen = photo
                used_urls.add(url)
                break
        if not chosen and photos:
            chosen = photos[0]
        p["image_url"] = chosen["src_large"] if chosen else ""
        p["image_preview"] = chosen["src_medium"] if chosen else ""


def run_content_agent(idea: str, market_summary: dict, website_content: dict,
                      customers: dict, context: dict = None) -> dict:
    print("\n[CONTENT AGENT] Starting...")
    context = context or {}
    website_content = website_content or {}
    brand = website_content.get("company_name") or context.get("startup_name") or "the business"
    city = context.get("city", "")
    out = {"brand": brand, "posts": [], "gmb_post": "", "whatsapp_announcement": "", "plan": {"weeks": []}}

    # ---- Call 1: social content ----
    try:
        prompt = (
            f"{build_context_string(idea, context)}\n"
            f"Brand name: {brand}\n"
            f"Tagline: {website_content.get('tagline', '')}\n"
            f"{market_brief(market_summary)}"
        )
        raw = call_groq(CONTENT_SYSTEM, prompt, max_tokens=4000)
        data = parse_json_response(raw)
        posts = data.get("posts", [])[:5]
        for p in posts:
            tags = p.get("hashtags", [])
            if isinstance(tags, str):
                tags = re.findall(r"#\w+", tags)
            p["hashtags"] = [("#" + t.lstrip("#")) for t in tags if str(t).strip()][:8]
            p["caption"] = str(p.get("caption", "")).strip()
        out["posts"] = posts
        out["gmb_post"] = str(data.get("gmb_post", "")).strip()
        out["whatsapp_announcement"] = str(data.get("whatsapp_announcement", "")).strip()

        # Attach one unique Pexels image per post, prepending the business type
        for p in posts:
            kw = str(p.get("image_keyword", "")).strip() or idea
            p["image_keyword"] = kw
        _fetch_post_images(posts, idea)
        print(f"[CONTENT AGENT] {len(posts)} posts written")
    except Exception as e:
        print(f"[CONTENT AGENT] Social content ERROR: {e}")
        out["content_error"] = str(e)

    # ---- Call 2: 30-day plan ----
    if LAUNCH_PLAN_ENABLED:
        time.sleep(3)  # pace Groq calls (8K tokens per minute limit)
        try:
            seg_names = [s.get("segment", "") for s in (customers or {}).get("segments", [])]
            post_types = [p.get("type", "") for p in out["posts"]]
            plan_prompt = (
                f"{build_context_string(idea, context)}\n"
                f"Brand name: {brand}\nCity: {city}\n"
                f"Week-one social posts: {', '.join(post_types) or 'not available'}\n"
                f"Customer groups to approach: {', '.join(seg_names) or 'not available'}\n"
                f"{market_brief(market_summary)}"
            )
            raw = call_groq(PLAN_SYSTEM, plan_prompt)
            plan = parse_json_response(raw)
            out["plan"] = {"weeks": plan.get("weeks", [])[:4]}
            print(f"[CONTENT AGENT] Plan has {len(out['plan']['weeks'])} weeks")
        except Exception as e:
            print(f"[CONTENT AGENT] Plan ERROR: {e}")
            out["plan_error"] = str(e)

    return out
