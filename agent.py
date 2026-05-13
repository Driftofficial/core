"""
Social listening agent for Larridin AI.

Monitors Reddit and Hacker News for posts where engineering leaders express
pain points that Larridin directly solves (AI sprawl, shadow AI, ROI, governance,
license waste, proficiency, security, standardization, and procurement).

Usage:
    python agent.py

Scheduled runs (every 6 hours via cron):
    0 */6 * * * cd /path/to/project && python agent.py >> agent.log 2>&1

Required environment variables (see .env.example):
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
    SLACK_WEBHOOK_URL
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and thresholds
# ---------------------------------------------------------------------------
SEEN_POSTS_FILE = Path("seen_posts.json")
LEADS_LOG_FILE = Path("leads_log.jsonl")

HIGH_THRESHOLD = 7.0
LOW_THRESHOLD = 4.0

NOW = int(time.time())
FOURTEEN_DAYS_SECS = 14 * 24 * 3600

# ---------------------------------------------------------------------------
# Target subreddits
# ---------------------------------------------------------------------------
SUBREDDITS = [
    "devops",
    "sysadmin",
    "ExperiencedDevs",
    "softwareengineering",
    "programming",
    "MachineLearning",
    "cscareerquestions",
    "ITManagers",
    "cto",
]

# Subreddits that raise the persona baseline score
LEADERSHIP_SUBREDDITS = {"ITManagers", "cto"}
DEVOPS_SUBREDDITS = {"devops", "sysadmin"}
IC_SUBREDDITS = {"cscareerquestions"}

# ---------------------------------------------------------------------------
# Search queries — one OR-joined query per cluster keeps API calls low
# (9 subreddits × 5 clusters = 45 Reddit calls; 5 HN calls)
# ---------------------------------------------------------------------------
SEARCH_CLUSTERS: dict[str, str] = {
    "sprawl_visibility": (
        '"AI tool sprawl" OR "shadow AI" OR "unauthorized AI" OR '
        '"AI governance" OR "track AI usage" OR "AI tool usage"'
    ),
    "roi_justification": (
        '"AI ROI" OR "justify AI spend" OR "measure AI productivity" OR '
        '"prove AI value" OR "Copilot ROI" OR "AI cost justification" OR '
        '"AI investment return"'
    ),
    "compliance_security": (
        '"AI compliance" OR "AI security risk" OR "data leakage AI" OR '
        '"AI policy enforcement" OR "AI audit" OR "AI procurement" OR '
        '"AI data leak"'
    ),
    "license_cost": (
        '"software license waste" OR "redundant AI" OR "duplicate AI tools" OR '
        '"AI license audit" OR "AI subscription cost" OR "AI tool costs out of control"'
    ),
    "proficiency_adoption": (
        '"AI adoption rate" OR "engineers not using AI" OR "AI proficiency" OR '
        '"low AI adoption" OR "AI training for engineers" OR "measuring AI skills"'
    ),
}

# Individual keywords for HN (Algolia handles them better than long OR strings)
HN_KEYWORDS: list[str] = [
    "AI tool sprawl",
    "shadow AI",
    "AI governance engineering",
    "AI ROI",
    "justify AI spend",
    "measure AI productivity",
    "AI compliance",
    "AI security risk",
    "data leakage AI",
    "AI audit",
    "software license waste",
    "redundant AI subscriptions",
    "AI adoption rate",
    "engineers not using AI",
    "AI proficiency",
]

# ---------------------------------------------------------------------------
# Pain point categories with keyword signals for detection
# ---------------------------------------------------------------------------
PAIN_CATEGORIES: dict[str, dict] = {
    "ai_tool_sprawl": {
        "label": "AI Tool Sprawl / Visibility",
        "signals": [
            "tool sprawl", "too many ai", "can't track", "no visibility into ai",
            "shadow ai", "unauthorized ai", "which ai tools", "track ai usage",
            "ai tool usage", "ai tools engineers", "unmanaged ai",
        ],
    },
    "shadow_ai": {
        "label": "Shadow AI",
        "signals": [
            "shadow ai", "unauthorized ai", "untracked ai", "using ai without",
            "without approval", "unsanctioned ai", "unapproved ai", "rogue ai",
        ],
    },
    "roi_measurement": {
        "label": "ROI / Business Value",
        "signals": [
            "ai roi", "return on investment", "justify ai", "prove ai value",
            "business value from ai", "measure ai productivity", "copilot roi",
            "ai investment", "ai spend", "ai cost justification", "ai investment return",
            "worth the cost", "paying for ai",
        ],
    },
    "compliance_governance": {
        "label": "Compliance / Governance",
        "signals": [
            "ai compliance", "ai governance", "ai policy", "ai regulation",
            "ai audit", "data leakage ai", "ai data leak", "gdpr ai", "hipaa ai",
            "sox ai", "policy enforcement", "data breach ai",
        ],
    },
    "license_waste": {
        "label": "License Waste / Cost Optimization",
        "signals": [
            "license waste", "redundant ai", "duplicate ai tool", "ai subscription cost",
            "ai tool cost", "ai license audit", "unused ai license", "overlapping ai",
            "paying for multiple", "ai tool overlap",
        ],
    },
    "ai_proficiency": {
        "label": "AI Proficiency / Adoption",
        "signals": [
            "ai proficiency", "ai adoption rate", "engineers not using ai",
            "low ai adoption", "ai training for engineer", "measuring ai skills",
            "ai skill gap", "developers not using", "team not adopting ai",
        ],
    },
    "leadership_pressure": {
        "label": "Leadership Pressure / Justification",
        "signals": [
            "leadership asking", "ceo wants", "board wants", "justify spend",
            "defend ai budget", "prove roi to", "executive pressure", "c-suite",
            "leadership is asking", "ceo is asking",
        ],
    },
    "security_risk": {
        "label": "Security Risk (Unmanaged AI)",
        "signals": [
            "ai security risk", "data leak", "sensitive data ai", "pii ai",
            "proprietary code ai", "code leak", "ip leak ai", "code going to",
            "training data", "sending code to", "data leaving",
        ],
    },
    "standardization": {
        "label": "AI Tooling Standardization",
        "signals": [
            "standardize ai", "standardization of ai", "which ai tool to use",
            "single ai tool", "consolidate ai", "fragmented ai tooling",
            "different teams using different", "ai tool decision",
        ],
    },
    "audit_procurement": {
        "label": "Audit / Procurement",
        "signals": [
            "ai procurement", "ai vendor", "ai tool audit", "procurement challenge",
            "vendor assessment ai", "ai tool evaluation", "evaluating ai tools",
            "ai vendor management",
        ],
    },
}

# ---------------------------------------------------------------------------
# Persona detection patterns (applied to lowercased post text)
# ---------------------------------------------------------------------------
LEADERSHIP_PATTERNS: list[str] = [
    r"\bcto\b",
    r"\bvp\s+of\s+engineering\b",
    r"\bvpe\b",
    r"\bdirector\s+of\s+engineering\b",
    r"\bengineering\s+director\b",
    r"\bengineering\s+manager\b",
    r"\bplatform\s+(team|lead|engineer)\b",
    r"\bdevops\s+lead\b",
    r"\binfrastructure\s+lead\b",
    r"\bi\s+(manage|lead)\s+(a|the|our|my)\s+(team|org|department|group)\b",
    r"\bmy\s+(team|engineers|org|department)\b",
    r"\bwe.re\s+evaluating\b",
    r"\bwe\s+are\s+evaluating\b",
    r"\bour\s+budget\b",
    r"\bi\s+own\s+the\b",
    r"\bi.m\s+responsible\s+for\b",
    r"\bteam\s+of\s+\d+\b",
    r"\b\d+\s+(engineers?|developers?|devs?)\b",
    r"\bhead\s+of\s+(engineering|platform|devops|infra)\b",
    r"\bstaff\s+engineer\b",
    r"\bprincipal\s+engineer\b",
]

URGENCY_SIGNALS: list[str] = [
    "frustrated", "struggling", "problem", "issue", "urgent", "concerned",
    "worried", "need to solve", "have to fix", "can't figure", "help me",
    "anyone else dealing", "we're stuck", "pain point", "nightmare",
    "out of control", "no idea", "overwhelmed",
]

AI_TOOL_NAMES: list[str] = [
    "copilot", "github copilot", "cursor", "codeium", "tabnine", "chatgpt",
    "claude", "gemini", "cody", "supermaven", "continue", "codex", "devin",
    "aws codewhisperer", "codewhisperer",
]

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def load_seen_posts() -> set[str]:
    if SEEN_POSTS_FILE.exists():
        try:
            return set(json.loads(SEEN_POSTS_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read seen_posts.json — starting fresh")
    return set()


def save_seen_posts(seen: set[str]) -> None:
    SEEN_POSTS_FILE.write_text(json.dumps(sorted(seen), indent=2))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def detect_pain_categories(text: str) -> list[str]:
    """Return list of matched Larridin pain category IDs."""
    text_lower = text.lower()
    return [
        cat_id
        for cat_id, cat in PAIN_CATEGORIES.items()
        if any(signal in text_lower for signal in cat["signals"])
    ]


def score_pain_specificity(text: str, categories: list[str]) -> float:
    """
    Score 1–10 based on how specific and compelling the pain point description is.
    More categories, specific tool names, urgency language, and detail all boost the score.
    """
    if not categories:
        return 1.0

    text_lower = text.lower()
    score = 2.0 + min(len(categories), 3) * 1.5  # 3.5 – 6.5 base range

    if any(u in text_lower for u in URGENCY_SIGNALS):
        score += 0.75

    if any(t in text_lower for t in AI_TOOL_NAMES):
        score += 0.5  # Named a specific AI tool — more context

    if re.search(r"\b\d+\b", text):
        score += 0.25  # Numbers suggest concrete detail

    if len(text) > 400:
        score += 0.25
    if len(text) > 1200:
        score += 0.25

    return min(round(score, 2), 10.0)


def score_persona_match(text: str, subreddit: Optional[str]) -> float:
    """
    Score 1–10 based on how likely the author is a Larridin ICP.
    Subreddit provides a baseline; leadership signals in the text can boost it.
    """
    if subreddit in LEADERSHIP_SUBREDDITS:
        base = 7.0
    elif subreddit in DEVOPS_SUBREDDITS:
        base = 5.5
    elif subreddit in IC_SUBREDDITS:
        base = 2.5
    elif subreddit is None:  # Hacker News
        base = 5.0
    else:
        base = 4.5

    text_lower = text.lower()
    leadership_matches = sum(
        1 for pattern in LEADERSHIP_PATTERNS if re.search(pattern, text_lower)
    )

    if leadership_matches >= 3:
        base = max(base, 9.0)
    elif leadership_matches >= 2:
        base = max(base, 8.0)
    elif leadership_matches >= 1:
        base = max(base, 6.5)

    return min(round(base, 2), 10.0)


def score_recency(posted_at_unix: int) -> float:
    """Score 1–10 based on post age; returns 0.0 if older than 14 days (discard)."""
    age_days = (NOW - posted_at_unix) / 86400
    if age_days < 1:
        return 10.0
    elif age_days <= 3:
        return 8.0
    elif age_days <= 7:
        return 6.0
    elif age_days <= 14:
        return 4.0
    return 0.0


def compute_final_score(pain: float, persona: float, recency: float) -> float:
    return round((pain + persona + recency) / 3, 2)


# ---------------------------------------------------------------------------
# Reddit client
# ---------------------------------------------------------------------------

class RedditClient:
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"
    API_BASE = "https://oauth.reddit.com"

    def __init__(self) -> None:
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.username = os.getenv("REDDIT_USERNAME", "")
        self.password = os.getenv("REDDIT_PASSWORD", "")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Larridin-SocialListening/1.0"
        self._authenticated = False

    def _authenticate(self) -> bool:
        if not all([self.client_id, self.client_secret, self.username, self.password]):
            log.warning("Reddit credentials not fully set — skipping Reddit")
            return False
        try:
            resp = self.session.post(
                self.AUTH_URL,
                auth=(self.client_id, self.client_secret),
                data={
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password,
                },
                timeout=15,
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]
            self.session.headers["Authorization"] = f"Bearer {token}"
            self._authenticated = True
            log.info("Reddit OAuth authenticated")
            return True
        except Exception as exc:
            log.error(f"Reddit authentication failed: {exc}")
            return False

    def _search_subreddit(self, subreddit: str, query: str) -> list[dict]:
        url = f"{self.API_BASE}/r/{subreddit}/search"
        params = {
            "q": query,
            "sort": "new",
            "t": "month",
            "limit": 25,
            "restrict_sr": 1,
            "type": "link",
        }
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            log.debug(f"Reddit search error r/{subreddit} — {exc}")
            return []

    def fetch_posts(self, seen: set[str]) -> list[dict]:
        if not self._authenticate():
            return []

        cutoff = NOW - FOURTEEN_DAYS_SECS
        results: list[dict] = []
        # Track raw Reddit IDs within this run to avoid per-cluster duplicates
        run_ids: set[str] = set()

        for subreddit in SUBREDDITS:
            for cluster_name, query in SEARCH_CLUSTERS.items():
                children = self._search_subreddit(subreddit, query)
                for child in children:
                    data = child.get("data", {})
                    raw_id = data.get("id", "")
                    post_id = f"reddit_{raw_id}"
                    if post_id in seen or raw_id in run_ids:
                        continue
                    run_ids.add(raw_id)

                    created = int(data.get("created_utc", 0))
                    if created < cutoff:
                        continue

                    title = data.get("title", "")
                    selftext = data.get("selftext", "")
                    full_text = f"{title}\n{selftext}".strip()
                    permalink = data.get("permalink", "")

                    results.append({
                        "id": post_id,
                        "source": "reddit",
                        "subreddit": subreddit,
                        "url": f"https://reddit.com{permalink}",
                        "username": f"u/{data.get('author', 'unknown')}",
                        "posted_at_unix": created,
                        "posted_at": datetime.fromtimestamp(
                            created, tz=timezone.utc
                        ).isoformat(),
                        "title": title,
                        "text": full_text,
                    })

                time.sleep(1.1)  # Reddit rate limit: ~1 req/sec for OAuth apps

        log.info(f"Reddit: fetched {len(results)} candidate posts")
        return results


# ---------------------------------------------------------------------------
# Hacker News client (Algolia search API)
# ---------------------------------------------------------------------------

class HNClient:
    SEARCH_URL = "https://hn.algolia.com/api/v1/search"

    def fetch_posts(self, seen: set[str]) -> list[dict]:
        cutoff = NOW - FOURTEEN_DAYS_SECS
        results: list[dict] = []
        run_ids: set[str] = set()

        for keyword in HN_KEYWORDS:
            params = {
                "query": keyword,
                "tags": "(story,comment)",
                "numericFilters": f"created_at_i>{cutoff}",
                "hitsPerPage": 20,
            }
            try:
                resp = requests.get(self.SEARCH_URL, params=params, timeout=15)
                resp.raise_for_status()
                hits = resp.json().get("hits", [])
                for hit in hits:
                    object_id = hit.get("objectID", "")
                    post_id = f"hn_{object_id}"
                    if post_id in seen or object_id in run_ids:
                        continue
                    run_ids.add(object_id)

                    created = int(hit.get("created_at_i", 0))
                    if created < cutoff:
                        continue

                    story_id = hit.get("story_id") or hit.get("objectID", "")
                    url = hit.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
                    title = hit.get("title") or hit.get("story_title") or ""
                    body = (hit.get("story_text") or "") + " " + (hit.get("comment_text") or "")
                    full_text = f"{title}\n{body}".strip()
                    author = hit.get("author", "unknown")

                    results.append({
                        "id": post_id,
                        "source": "hackernews",
                        "subreddit": None,
                        "url": url,
                        "username": author,
                        "posted_at_unix": created,
                        "posted_at": datetime.fromtimestamp(
                            created, tz=timezone.utc
                        ).isoformat(),
                        "title": title,
                        "text": full_text,
                    })
            except Exception as exc:
                log.debug(f"HN search error '{keyword}': {exc}")

            time.sleep(0.5)  # Be polite to the Algolia API

        log.info(f"HN: fetched {len(results)} candidate posts")
        return results


# ---------------------------------------------------------------------------
# Slack alerting
# ---------------------------------------------------------------------------

def _build_persona_assessment(text: str, subreddit: Optional[str]) -> str:
    text_lower = text.lower()
    matched_patterns = [p for p in LEADERSHIP_PATTERNS if re.search(p, text_lower)]
    if matched_patterns:
        # Extract a snippet of the matching text for context
        for p in matched_patterns[:2]:
            m = re.search(p, text_lower)
            if m:
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 30)
                snippet = text[start:end].strip().replace("\n", " ")
                return (
                    f"Post contains leadership signal: '...{snippet}...'. "
                    "Likely an engineering leader or technical decision-maker."
                )
    if subreddit in LEADERSHIP_SUBREDDITS:
        return (
            f"Posted in r/{subreddit}, frequented by IT and engineering managers. "
            "Specific role not confirmed from post text alone."
        )
    if subreddit in DEVOPS_SUBREDDITS:
        return (
            f"Posted in r/{subreddit}. May be a DevOps/platform lead or IC. "
            "Manual enrichment recommended to confirm seniority."
        )
    return "Role and seniority unclear from post text. Manual enrichment strongly recommended."


def send_slack_alert(
    post: dict,
    categories: list[str],
    score: float,
    pain_score: float,
    persona_score: float,
    recency_score: float,
) -> None:
    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    user_id = os.getenv("SLACK_USER_ID", "")
    if not bot_token or not user_id:
        log.warning("SLACK_BOT_TOKEN or SLACK_USER_ID not set — Slack alert skipped")
        return

    age_days = (NOW - post["posted_at_unix"]) / 86400
    if age_days < 1:
        age_str = "today"
    elif age_days < 2:
        age_str = "1 day ago"
    else:
        age_str = f"{int(age_days)} days ago"

    source_label = (
        f"Reddit — r/{post['subreddit']}" if post["source"] == "reddit" else "Hacker News"
    )
    category_labels = ", ".join(
        PAIN_CATEGORIES[c]["label"] for c in categories if c in PAIN_CATEGORIES
    ) or "General AI Pain Point"

    # Truncate post text to a safe summary length
    summary_text = post["text"][:600].replace("\n", " ").strip()
    if len(post["text"]) > 600:
        summary_text += "..."

    why_match = (
        f"Directly expresses pain around {category_labels.lower()}, "
        "mapping to Larridin's core capabilities in AI visibility, governance, and ROI measurement."
    )
    persona_assessment = _build_persona_assessment(post["text"], post.get("subreddit"))

    message = (
        f":rotating_light: *Larridin Signal Detected*\n\n"
        f"*Source:* {source_label} | <{post['url']}|View Post>\n"
        f"*Posted:* {age_str}\n"
        f"*Score:* {score} / 10  "
        f"_(Pain: {pain_score:.1f} | Persona: {persona_score:.1f} | Recency: {recency_score:.1f})_\n\n"
        f"*Pain Point Category:* {category_labels}\n\n"
        f"*Post Summary:*\n{summary_text}\n\n"
        f"*Why It Matches Larridin:*\n{why_match}\n\n"
        f"*Persona Assessment:*\n{persona_assessment}\n\n"
        f"*Username:* {post['username']}\n"
        f"*Manual Enrichment Needed:* Cross-reference username on LinkedIn and Google. "
        f"Look for posts in their history that reveal company, name, or role."
    )

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {bot_token}"},
            json={"channel": user_id, "text": message},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            log.error(f"Slack API error for {post['id']}: {payload.get('error')}")
        else:
            log.info(f"Slack DM sent: {post['id']}")
    except Exception as exc:
        log.error(f"Slack DM failed for {post['id']}: {exc}")


# ---------------------------------------------------------------------------
# Lead logging
# ---------------------------------------------------------------------------

def append_lead(
    post: dict,
    categories: list[str],
    score: float,
    tier: str,
    pain_score: float,
    persona_score: float,
    recency_score: float,
) -> None:
    record = {
        "id": post["id"],
        "source": post["source"],
        "subreddit": post.get("subreddit"),
        "url": post["url"],
        "username": post["username"],
        "posted_at": post["posted_at"],
        "score": score,
        "pain_score": pain_score,
        "persona_score": persona_score,
        "recency_score": recency_score,
        "tier": tier,
        "pain_categories": categories,
        "summary": post["text"][:500].strip(),
        "persona_assessment": _build_persona_assessment(
            post["text"], post.get("subreddit")
        ),
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    with LEADS_LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    log.info(f"Logged {tier} lead: {post['id']} (score={score})")


# ---------------------------------------------------------------------------
# Post processing pipeline
# ---------------------------------------------------------------------------

def process_posts(posts: list[dict], seen: set[str]) -> tuple[int, int, int]:
    """Score and route posts. Returns (high_count, low_count, discarded_count)."""
    high_count = low_count = discarded = 0

    for post in posts:
        post_id = post["id"]

        recency = score_recency(post["posted_at_unix"])
        seen.add(post_id)

        if recency == 0.0:
            discarded += 1
            continue  # Older than 14 days — discard

        categories = detect_pain_categories(post["text"])
        if not categories:
            discarded += 1
            continue  # No relevant pain point detected

        pain = score_pain_specificity(post["text"], categories)
        persona = score_persona_match(post["text"], post.get("subreddit"))
        final = compute_final_score(pain, persona, recency)

        if final >= HIGH_THRESHOLD:
            send_slack_alert(post, categories, final, pain, persona, recency)
            append_lead(post, categories, final, "high", pain, persona, recency)
            high_count += 1
        elif final >= LOW_THRESHOLD:
            append_lead(post, categories, final, "low", pain, persona, recency)
            low_count += 1
        else:
            log.debug(f"Discarded {post_id} — score {final} below threshold")
            discarded += 1

    return high_count, low_count, discarded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    log.info("=== Larridin Social Listening Agent — run started ===")

    seen = load_seen_posts()
    log.info(f"Loaded {len(seen)} previously seen post IDs")

    reddit_posts = RedditClient().fetch_posts(seen)
    hn_posts = HNClient().fetch_posts(seen)
    all_posts = reddit_posts + hn_posts
    log.info(f"Total new candidates: {len(all_posts)}")

    high, low, discarded = process_posts(all_posts, seen)

    save_seen_posts(seen)
    log.info(
        f"Run complete — high={high}, low={low}, discarded={discarded}, "
        f"total_seen={len(seen)}"
    )


if __name__ == "__main__":
    run()
