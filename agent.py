"""
Social listening agent for Larridin AI.

Monitors Reddit (public JSON API — no credentials required) and Hacker News
(Algolia API) for engineering leaders expressing pain points that Larridin
directly solves: AI sprawl, shadow AI, ROI, governance, license waste,
proficiency gaps, security risk, standardization, and procurement challenges.

Alerts are delivered as Telegram DMs.

Usage:
    python agent.py                    # full run
    python agent.py --dry-run          # score & log without sending messages
    python agent.py --source reddit    # Reddit only
    python agent.py --source hn        # Hacker News only

Scheduled runs (every 6 hours via cron):
    0 */6 * * * cd /path/to/project && python agent.py >> agent.log 2>&1

Required environment variables (see .env.example):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

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
# Module-level run flags (set by main() before any processing)
# ---------------------------------------------------------------------------
DRY_RUN: bool = False

# ---------------------------------------------------------------------------
# Paths and thresholds
# ---------------------------------------------------------------------------
SEEN_POSTS_FILE = Path("seen_posts.json")
LEADS_LOG_FILE = Path("leads_log.jsonl")

MAX_TEXT_LEN = 10_000

HIGH_THRESHOLD = 7.0
LOW_THRESHOLD = 4.0

NOW = int(time.time())
FOURTEEN_DAYS_SECS = 14 * 24 * 3600

# ---------------------------------------------------------------------------
# Accounts to skip at ingest
# ---------------------------------------------------------------------------
BLOCKED_AUTHORS: set[str] = {
    "AutoModerator", "automoderator", "[deleted]", "[removed]",
    "BotDefense", "RepostSleuthBot", "RemindMeBot", "sneakpeekbot",
}

# ---------------------------------------------------------------------------
# Target subreddits
# ---------------------------------------------------------------------------
SUBREDDITS: list[str] = [
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

LEADERSHIP_SUBREDDITS: set[str] = {"ITManagers", "cto"}
DEVOPS_SUBREDDITS: set[str] = {"devops", "sysadmin"}
IC_SUBREDDITS: set[str] = {"cscareerquestions"}

# ---------------------------------------------------------------------------
# Search queries — one OR-joined query per cluster
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
# Pain point categories
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
# Persona detection
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
# Security helpers
# ---------------------------------------------------------------------------

_PRIVATE_IP_RE = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)$",
    re.IGNORECASE,
)

_REDDIT_PERMALINK_RE = re.compile(
    r"^/r/[A-Za-z0-9_]{1,50}/comments/[A-Za-z0-9_]+/"
)


def _reddit_url_from_permalink(permalink: str) -> Optional[str]:
    if not _REDDIT_PERMALINK_RE.match(permalink):
        log.warning(f"Unexpected Reddit permalink shape: {permalink!r}")
        return None
    return f"https://www.reddit.com{permalink}"


def _hn_item_url(object_id: str) -> Optional[str]:
    if not re.match(r"^\d{1,15}$", str(object_id)):
        log.warning(f"Non-numeric HN objectID: {object_id!r}")
        return None
    return f"https://news.ycombinator.com/item?id={object_id}"


def _html_escape(text: str) -> str:
    """Escape HTML special characters in externally sourced text (for Telegram HTML mode)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    return text[:max_len] if len(text) > max_len else text


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
    text_lower = text.lower()
    return [
        cat_id
        for cat_id, cat in PAIN_CATEGORIES.items()
        if any(signal in text_lower for signal in cat["signals"])
    ]


def score_pain_specificity(text: str, categories: list[str]) -> float:
    if not categories:
        return 1.0

    text_lower = text.lower()
    score = 2.0 + min(len(categories), 3) * 1.5

    if any(u in text_lower for u in URGENCY_SIGNALS):
        score += 0.75
    if any(t in text_lower for t in AI_TOOL_NAMES):
        score += 0.5
    if re.search(r"\b\d+\b", text):
        score += 0.25
    if len(text) > 400:
        score += 0.25
    if len(text) > 1200:
        score += 0.25

    return min(round(score, 2), 10.0)


def score_persona_match(text: str, subreddit: Optional[str]) -> float:
    if subreddit in LEADERSHIP_SUBREDDITS:
        base = 7.0
    elif subreddit in DEVOPS_SUBREDDITS:
        base = 5.5
    elif subreddit in IC_SUBREDDITS:
        base = 2.5
    elif subreddit is None:
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
# Reddit client — public JSON API, no credentials required
# ---------------------------------------------------------------------------

class RedditClient:
    BASE = "https://www.reddit.com"
    # Reddit requires a descriptive User-Agent for public API access
    HEADERS = {"User-Agent": "Larridin-SocialListening/1.0 (social listening agent)"}

    def _search(self, subreddit: str, query: str, content_type: str) -> list[dict]:
        url = f"{self.BASE}/r/{subreddit}/search.json"
        params = {
            "q": query,
            "sort": "new",
            "t": "month",
            "limit": 25,
            "restrict_sr": 1,
            "type": content_type,
        }
        try:
            resp = requests.get(url, params=params, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            log.debug(f"Reddit search error r/{subreddit} ({content_type}): {exc}")
            return []

    def _parse_post(self, data: dict, subreddit: str, cutoff: int) -> Optional[dict]:
        author = data.get("author", "")
        if author in BLOCKED_AUTHORS:
            return None
        try:
            created = int(data.get("created_utc", 0))
        except (TypeError, ValueError):
            return None
        if created < cutoff:
            return None

        permalink = data.get("permalink", "")
        url = _reddit_url_from_permalink(permalink)
        if not url:
            return None

        title = str(data.get("title", ""))[:500]
        selftext = str(data.get("selftext", ""))
        full_text = _truncate(f"{title}\n{selftext}".strip())

        return {
            "id": f"reddit_{data.get('id', '')}",
            "source": "reddit",
            "content_type": "post",
            "subreddit": subreddit,
            "url": url,
            "username": f"u/{author}",
            "posted_at_unix": created,
            "posted_at": datetime.fromtimestamp(created, tz=timezone.utc).isoformat(),
            "title": title,
            "text": full_text,
        }

    def _parse_comment(self, data: dict, subreddit: str, cutoff: int) -> Optional[dict]:
        author = data.get("author", "")
        if author in BLOCKED_AUTHORS:
            return None
        try:
            created = int(data.get("created_utc", 0))
        except (TypeError, ValueError):
            return None
        if created < cutoff:
            return None

        permalink = data.get("permalink", "")
        url = _reddit_url_from_permalink(permalink)
        if not url:
            return None

        body = _truncate(str(data.get("body", "")))
        link_title = str(data.get("link_title", ""))[:500]
        full_text = f"{link_title}\n{body}".strip() if link_title else body

        return {
            "id": f"reddit_{data.get('id', '')}",
            "source": "reddit",
            "content_type": "comment",
            "subreddit": subreddit,
            "url": url,
            "username": f"u/{author}",
            "posted_at_unix": created,
            "posted_at": datetime.fromtimestamp(created, tz=timezone.utc).isoformat(),
            "title": link_title,
            "text": full_text,
        }

    def fetch_posts(self, seen: set[str]) -> list[dict]:
        cutoff = NOW - FOURTEEN_DAYS_SECS
        results: list[dict] = []
        run_ids: set[str] = set()

        for subreddit in SUBREDDITS:
            for cluster_name, query in SEARCH_CLUSTERS.items():
                for content_type, parser in [
                    ("link", self._parse_post),
                    ("comment", self._parse_comment),
                ]:
                    children = self._search(subreddit, query, content_type)
                    for child in children:
                        data = child.get("data", {})
                        raw_id = data.get("id", "")
                        post_id = f"reddit_{raw_id}"
                        if post_id in seen or raw_id in run_ids:
                            continue

                        record = parser(data, subreddit, cutoff)
                        if record is None:
                            continue

                        run_ids.add(raw_id)
                        results.append(record)

                    # Public API: 1 req/2 sec to stay well within rate limits
                    time.sleep(2.0)

        log.info(f"Reddit: fetched {len(results)} candidate posts/comments")
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
                    object_id = str(hit.get("objectID", ""))
                    post_id = f"hn_{object_id}"
                    if post_id in seen or object_id in run_ids:
                        continue

                    author = hit.get("author", "unknown")
                    if author in BLOCKED_AUTHORS:
                        continue

                    try:
                        created = int(hit.get("created_at_i", 0))
                    except (TypeError, ValueError):
                        continue
                    if created < cutoff:
                        continue

                    url = _hn_item_url(object_id)
                    if not url:
                        continue

                    title = str(hit.get("title") or hit.get("story_title") or "")[:500]
                    body = (
                        str(hit.get("story_text") or "")
                        + " "
                        + str(hit.get("comment_text") or "")
                    ).strip()
                    full_text = _truncate(f"{title}\n{body}".strip())

                    run_ids.add(object_id)
                    results.append({
                        "id": post_id,
                        "source": "hackernews",
                        "content_type": "story" if hit.get("title") else "comment",
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

            time.sleep(0.5)

        log.info(f"HN: fetched {len(results)} candidate posts/comments")
        return results


# ---------------------------------------------------------------------------
# Persona assessment
# ---------------------------------------------------------------------------

def _build_persona_assessment(text: str, subreddit: Optional[str]) -> str:
    text_lower = text.lower()
    for pattern in LEADERSHIP_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 30)
            snippet = text[start:end].strip().replace("\n", " ")
            return (
                f"Leadership signal detected: '...{snippet}...'. "
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
            "Manual enrichment recommended."
        )
    return "Role unclear from post text. Manual enrichment strongly recommended."


# ---------------------------------------------------------------------------
# Telegram alerting
# ---------------------------------------------------------------------------

def send_telegram_alert(
    post: dict,
    categories: list[str],
    score: float,
    pain_score: float,
    persona_score: float,
    recency_score: float,
) -> None:
    if DRY_RUN:
        log.info(f"[DRY RUN] Would send Telegram message for {post['id']} (score={score})")
        return

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        log.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — alert skipped")
        return

    age_days = (NOW - post["posted_at_unix"]) / 86400
    if age_days < 1:
        age_str = "today"
    elif age_days < 2:
        age_str = "1 day ago"
    else:
        age_str = f"{int(age_days)} days ago"

    source_label = (
        f"Reddit — r/{post['subreddit']} ({post.get('content_type', 'post')})"
        if post["source"] == "reddit"
        else f"Hacker News ({post.get('content_type', 'story')})"
    )
    category_labels = ", ".join(
        PAIN_CATEGORIES[c]["label"] for c in categories if c in PAIN_CATEGORIES
    ) or "General AI Pain Point"

    # Escape external content before inserting into HTML
    summary_raw = post["text"][:400].replace("\n", " ").strip()
    if len(post["text"]) > 400:
        summary_raw += "..."
    summary = _html_escape(summary_raw)

    persona = _html_escape(_build_persona_assessment(post["text"], post.get("subreddit")))
    username = _html_escape(post["username"])
    source_escaped = _html_escape(source_label)
    categories_escaped = _html_escape(category_labels)

    why_match = (
        f"Directly expresses pain around {_html_escape(category_labels.lower())}, "
        "mapping to Larridin's capabilities in AI visibility, governance, and ROI measurement."
    )

    # Telegram HTML mode — keep under 4096 chars
    message = (
        f"🚨 <b>Larridin Signal Detected</b>\n\n"
        f"<b>Source:</b> {source_escaped}\n"
        f"<b>Posted:</b> {age_str}\n"
        f"<b>Score:</b> {score} / 10  "
        f"<i>(Pain: {pain_score:.1f} | Persona: {persona_score:.1f} | Recency: {recency_score:.1f})</i>\n\n"
        f"<b>Pain Point:</b> {categories_escaped}\n\n"
        f"<b>Summary:</b>\n{summary}\n\n"
        f"<b>Why It Matches Larridin:</b>\n{why_match}\n\n"
        f"<b>Persona:</b>\n{persona}\n\n"
        f"<b>Username:</b> {username}\n"
        f"<b>Enrichment:</b> Search LinkedIn &amp; Google for company, name, or role\n\n"
        f'<a href="{post["url"]}">View Post →</a>'
    )

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            log.error(f"Telegram error for {post['id']}: {payload.get('description')}")
        else:
            log.info(f"Telegram message sent: {post['id']}")
    except Exception as exc:
        log.error(f"Telegram send failed for {post['id']}: {type(exc).__name__}")


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
    if DRY_RUN:
        log.info(f"[DRY RUN] Would log {tier} lead: {post['id']} (score={score})")
        return

    record = {
        "id": post["id"],
        "source": post["source"],
        "content_type": post.get("content_type"),
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
# Processing pipeline
# ---------------------------------------------------------------------------

def process_posts(posts: list[dict], seen: set[str]) -> tuple[int, int, int]:
    high_count = low_count = discarded = 0

    for post in posts:
        post_id = post["id"]
        seen.add(post_id)

        recency = score_recency(post["posted_at_unix"])
        if recency == 0.0:
            discarded += 1
            continue

        categories = detect_pain_categories(post["text"])
        if not categories:
            discarded += 1
            continue

        pain = score_pain_specificity(post["text"], categories)
        persona = score_persona_match(post["text"], post.get("subreddit"))
        final = compute_final_score(pain, persona, recency)

        if final >= HIGH_THRESHOLD:
            send_telegram_alert(post, categories, final, pain, persona, recency)
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

def run(source: str = "all") -> None:
    log.info(
        f"=== Larridin Social Listening Agent — "
        f"source={source} dry_run={DRY_RUN} ==="
    )

    seen = load_seen_posts()
    log.info(f"Loaded {len(seen)} previously seen post IDs")

    reddit_posts: list[dict] = []
    hn_posts: list[dict] = []

    if source in ("all", "reddit"):
        reddit_posts = RedditClient().fetch_posts(seen)
    if source in ("all", "hn"):
        hn_posts = HNClient().fetch_posts(seen)

    all_posts = reddit_posts + hn_posts
    log.info(f"Total new candidates: {len(all_posts)}")

    high, low, discarded = process_posts(all_posts, seen)

    if not DRY_RUN:
        save_seen_posts(seen)

    log.info(
        f"Run complete — high={high}, low={low}, discarded={discarded}, "
        f"total_seen={len(seen)}"
    )


def main() -> None:
    global DRY_RUN

    parser = argparse.ArgumentParser(
        description="Larridin social listening agent"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score posts but skip Telegram alerts and file writes",
    )
    parser.add_argument(
        "--source",
        choices=["all", "reddit", "hn"],
        default="all",
        help="Data source to query (default: all)",
    )
    args = parser.parse_args()

    DRY_RUN = args.dry_run
    if DRY_RUN:
        log.info("Dry-run mode — no Telegram messages or file writes will occur")

    run(source=args.source)


if __name__ == "__main__":
    main()
