import os
import json
import time
import requests as req
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

load_dotenv()

# --- LLM (Groq) ---
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def call_groq(system_prompt: str, user_prompt: str, retries: int = 3, max_tokens: int = 2500) -> str:
    """Send a prompt to Groq with retry logic for rate limits."""
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                wait_time = 15 * (attempt + 1)
                print(f"[GROQ] Rate limited. Waiting {wait_time}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Groq rate limit exceeded after all retries.")


def parse_json_response(raw_text: str) -> dict:
    """
    Groq models sometimes wrap JSON in markdown code fences or add
    extra text before or after it. This strips that and parses safely.
    """
    text = raw_text.strip()
    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{") or cleaned.startswith("["):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model response.\nRaw: {raw_text[:500]}")


# --- Web Search (Tavily) ---
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily."""
    response = tavily_client.search(query=query, max_results=max_results)
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return results


# --- Image Search (Pexels) ---
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")


def search_images(query: str, per_page: int = 3) -> list[dict]:
    """
    Search Pexels for photos. Returns list of dicts with
    'url', 'photographer', 'alt', 'src_large', 'src_medium'.
    """
    if not PEXELS_API_KEY:
        print("[PEXELS] WARNING: No PEXELS_API_KEY found in .env")
        return []
    try:
        resp = req.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[PEXELS] Error: HTTP {resp.status_code}")
            return []
        data = resp.json()
        photos = []
        for p in data.get("photos", []):
            photos.append({
                "url": p.get("url", ""),
                "photographer": p.get("photographer", ""),
                "alt": p.get("alt", query),
                "src_large": p.get("src", {}).get("large2x", ""),
                "src_medium": p.get("src", {}).get("large", ""),
                "src_small": p.get("src", {}).get("medium", ""),
            })
        print(f"[PEXELS] Found {len(photos)} photos for '{query}'")
        return photos
    except Exception as e:
        print(f"[PEXELS] ERROR: {e}")
        return []


# =============================================================
# BUFFER (GraphQL API, https://api.buffer.com)
# Query, mutation and field names below were checked against
# developers.buffer.com (reference.md and the Posts & Scheduling guide).
# NOT verified: how many seconds Instagram takes to publish.
# Run buffer_test.py once to measure it on your own account.
# =============================================================
BUFFER_API_URL = "https://api.buffer.com"


class BufferError(Exception):
    pass


def _buffer_key() -> str:
    key = os.environ.get("BUFFER_API_KEY", "")
    if not key:
        raise BufferError("BUFFER_API_KEY is not set in .env")
    return key


def buffer_graphql(query: str, variables: dict | None = None) -> dict:
    """Run one GraphQL request. Returns the 'data' dict or raises BufferError."""
    try:
        resp = req.post(
            BUFFER_API_URL,
            headers={
                "Authorization": f"Bearer {_buffer_key()}",
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables or {}},
            timeout=25,
        )
    except req.RequestException as e:
        raise BufferError(f"Could not reach Buffer: {e}")

    if resp.status_code == 429:
        raise BufferError("Buffer rate limit reached (HTTP 429). Try again later.")
    try:
        body = resp.json()
    except ValueError:
        raise BufferError(f"Buffer returned HTTP {resp.status_code} with a non-JSON body.")

    if body.get("errors"):
        msg = "; ".join(str(e.get("message", e)) for e in body["errors"])
        raise BufferError(f"Buffer API error: {msg}")
    if resp.status_code >= 400:
        raise BufferError(f"Buffer returned HTTP {resp.status_code}.")
    return body.get("data") or {}


def buffer_list_channels() -> list[dict]:
    """List every channel in every organization on the account (used for setup)."""
    data = buffer_graphql("query { account { organizations { id name } } }")
    channels = []
    for org in data.get("account", {}).get("organizations", []):
        d = buffer_graphql(
            "query GetChannels($input: ChannelsInput!) {"
            " channels(input: $input) { id name service displayName } }",
            {"input": {"organizationId": org["id"]}},
        )
        for ch in d.get("channels", []):
            ch["organization"] = org.get("name", "")
            channels.append(ch)
    return channels


def buffer_get_channel(channel_id: str) -> dict:
    data = buffer_graphql(
        "query GetChannel($input: ChannelInput!) {"
        " channel(input: $input) { id name service displayName } }",
        {"input": {"id": channel_id}},
    )
    return data.get("channel") or {}


_CREATE_POST = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess { post { id status } }
    ... on MutationError { message }
  }
}
"""


def buffer_publish_now(channel_id: str, text: str, image_url: str) -> dict:
    """
    Publish one image post immediately (mode: shareNow, automatic publishing).
    Returns {"id": ..., "status": ...}. Raises BufferError on failure.
    """
    base_input = {
        "channelId": channel_id,
        "text": text,
        "schedulingType": "automatic",
        "mode": "shareNow",
        "assets": [{"image": {"url": image_url}}],
    }
    # First try with explicit Instagram metadata. If Buffer rejects that shape,
    # retry once without it (metadata is optional in the schema).
    with_meta = dict(base_input)
    with_meta["metadata"] = {"instagram": {"type": "post", "shouldShareToFeed": True}}

    last_error = ""
    for attempt_input in (with_meta, base_input):
        try:
            data = buffer_graphql(_CREATE_POST, {"input": attempt_input})
        except BufferError as e:
            last_error = str(e)
            continue
        result = data.get("createPost") or {}
        post = result.get("post")
        if post:
            return {"id": post.get("id"), "status": post.get("status")}
        last_error = result.get("message", "Unknown Buffer error")
    raise BufferError(last_error or "Buffer did not create the post.")


def buffer_get_post(post_id: str) -> dict:
    """Return {"id", "status", "sentAt", "externalLink", "error"} for one post."""
    data = buffer_graphql(
        "query GetPost($input: PostInput!) {"
        " post(input: $input) { id status sentAt externalLink error { message } } }",
        {"input": {"id": post_id}},
    )
    post = data.get("post") or {}
    err = post.get("error") or {}
    return {
        "id": post.get("id"),
        "status": str(post.get("status") or "").lower(),
        "sentAt": post.get("sentAt"),
        "externalLink": post.get("externalLink"),
        "error": err.get("message") if isinstance(err, dict) else None,
    }


def buffer_delete_post(post_id: str) -> bool:
    data = buffer_graphql(
        "mutation DeletePost($input: DeletePostInput!) {"
        " deletePost(input: $input) {"
        " ... on DeletePostSuccess { id } ... on MutationError { message } } }",
        {"input": {"id": post_id}},
    )
    return bool((data.get("deletePost") or {}).get("id"))
