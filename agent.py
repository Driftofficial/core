"""
Social listening agent for Larridin AI.

Monitors Reddit (posts + comments, 9 subreddits via OAuth) and Hacker News
(Algolia API) for engineering leaders expressing pain points that Larridin
directly solves: AI sprawl, shadow AI, ROI, governance, license waste,
proficiency gaps, security risk, standardization, and procurement challenges.

Usage:
    python agent.py                    # full run
    python agent.py --dry-run          # score & log without sending to Slack
    python agent.py --source reddit    # Reddit only
    python agent.py --source hn        # Hacker News only

Scheduled runs (every 6 hours via cron):
    0 */6 * * * cd /path/to/project && python agent.py >> agent.log 2>&1

Required environment variables (see .env.example):
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
    SLACK_BOT_TOKEN, SLACK_USER_ID
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

# Ingest cap — prevents pathological API responses from exhausting memory
MAX_TEXT_LEN = 10_000

HIGH_THRESHOLD = 7.0
LOW_THRESHOLD = 4.0

NOW = int(time.time())
FOURTEEN_DAYS_SECS = 14 * 24 * 3600

# ---------------------------------------------------------------------------
# Accounts to skip at ingest (bots, deleted users, auto-moderation)
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
# (9 subreddits × 5 clusters × 2 types (post+comment) = 90 Reddit API calls)
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

# Individual keywords for HN Algolia (handles focused terms better than OR chains)
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

# Private/reserved IP ranges that must never appear in outbound URLs
_PRIVATE_IP_RE = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)$",
    re.IGNORECASE,
)


def _safe_url(url: str, allowed_hosts: Optional[set[str]] = None) -> Optional[str]:
    """
    Validate that a URL is safe to embed in a Slack message or log.

    Returns the URL unchanged if it passes, or None if it should be rejected.
    Rejects: non-https schemes, private/localhost hosts, and (optionally)
    hosts not in allowed_hosts.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    if parsed.scheme != "https":
        log.debug(f"Rejected non-https URL: scheme={parsed.scheme!r}")
        return None

    host = parsed.hostname or ""
    if not host:
        return None

    if _PRIVATE_IP_RE.match(host):
        log.warning(f"Rejected private-host URL: {host!r}")
        return None

    if allowed_hosts and host not in allowed_hosts:
        log.debug(f"URL host {host!r} not in allowlist — rejected")
        return None

    return url


_REDDIT_PERMALINK_RE = re.compile(
    r"^/r/[A-Za-z0-9_]{1,50}/comments/[A-Za-z0-9_]+/"
)


def _reddit_url_from_permalink(permalink: str) -> Optional[str]:
    """Convert a Reddit API permalink to a safe absolute URL."""
    if not _REDDIT_PERMALINK_RE.match(permalink):
        log.warning(f"Unexpected Reddit permalink shape: {permalink!r}")
        return None
    return f"https://www.reddit.com{permalink}"


def _hn_item_url(object_id: str) -> Optional[str]:
    """Return a safe HN item URL from a numeric object ID."""
    if not re.match(r"^\d{1,15}$", str(object_id)):
        log.warning(f"Non-numeric HN objectID: {object_id!r}")
        return None
    return f"https://news.ycombinator.com/item?id={object_id}"


def _slack_escape(text: str) -> str:
    """Escape Slack mrkdwn special characters in externally sourced text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    """Hard-cap text at max_len characters to prevent memory abuse."""
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
    score = 2.0 + min(len(categories), 3) * 1.5  # 3.5 – 6.5 base

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
        except requests.HTTPError as exc:
            # Log the status code only — never log response body which may echo credentials
            log.error(f"Reddit authentication failed: HTTP {exc.response.status_code}")
            return False
        except Exception:
            log.error("Reddit authentication failed: network or parse error")
            return False

    def _search(self, subreddit: str, query: str, content_type: str) -> list[dict]:
        """
        Search a subreddit for posts or comments matching query.
        content_type: "link" for posts, "comment" for comments.
        """
        url = f"{self.API_BASE}/r/{subreddit}/search"
        params = {
            "q": query,
            "sort": "new",
            "t": "month",
            "limit": 25,
            "restrict_sr": 1,
            "type": content_type,
        }
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            log.debug(f"Reddit search error r/{subreddit} ({content_type}): {exc}")
            return []

    def _parse_post(self, data: dict, subreddit: str, cutoff: int) -> Optional[dict]:
        """Parse a Reddit post (link) into a normalised record, or None to skip."""
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
        """Parse a Reddit comment into a normalised record, or None to skip."""
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
        if not self._authenticate():
            return []

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

                    time.sleep(1.1)  # Reddit rate limit: ~1 req/sec for OAuth apps

        log.info(f"Reddit: fetched {len(results)} candidate posts/comments")
        return results


# ---------------------------------------------------------------------------
# Hacker News client (Algolia search API)
# ---------------------------------------------------------------------------

_HN_ALLOWED_HOST = "news.ycombinator.com"


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

                    # Always use the canonical HN item URL — never trust story URLs
                    # from user-submitted content as the primary link
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
# Persona assessment (shared between Slack and log)
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


# ---------------------------------------------------------------------------
# Slack alerting (Block Kit)
# ---------------------------------------------------------------------------

def _build_slack_blocks(
    post: dict,
    categories: list[str],
    score: float,
    pain_score: float,
    persona_score: float,
    recency_score: float,
) -> list[dict]:
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

    # Escape external content before inserting into Slack mrkdwn
    summary_raw = post["text"][:600].replace("\n", " ").strip()
    if len(post["text"]) > 600:
        summary_raw += "..."
    summary = _slack_escape(summary_raw)

    persona = _slack_escape(_build_persona_assessment(post["text"], post.get("subreddit")))
    username = _slack_escape(post["username"])

    why_match = (
        f"Directly expresses pain around {_slack_escape(category_labels.lower())}, "
        "mapping to Larridin's capabilities in AI visibility, governance, and ROI measurement."
    )

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🚨 Larridin Signal Detected"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Source*\n{_slack_escape(source_label)}"},
                {"type": "mrkdwn", "text": f"*Posted*\n{age_str}"},
                {"type": "mrkdwn", "text": f"*Score*\n{score} / 10"},
                {"type": "mrkdwn", "text": f"*Subscores*\nPain {pain_score:.1f} | Persona {persona_score:.1f} | Recency {recency_score:.1f}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Pain Point Category*\n{_slack_escape(category_labels)}",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Post Summary*\n{summary}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Why It Matches Larridin*\n{why_match}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Persona Assessment*\n{persona}"},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Username:* {username}  |  "
                        "*Enrichment:* Search LinkedIn & Google for company, name, or role"
                    ),
                }
            ],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Post"},
                    "url": post["url"],
                    "style": "primary",
                }
            ],
        },
    ]


def send_slack_alert(
    post: dict,
    categories: list[str],
    score: float,
    pain_score: float,
    persona_score: float,
    recency_score: float,
) -> None:
    if DRY_RUN:
        log.info(f"[DRY RUN] Would send Slack DM for {post['id']} (score={score})")
        return

    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    user_id = os.getenv("SLACK_USER_ID", "")
    if not bot_token or not user_id:
        log.warning("SLACK_BOT_TOKEN or SLACK_USER_ID not set — Slack alert skipped")
        return

    blocks = _build_slack_blocks(
        post, categories, score, pain_score, persona_score, recency_score
    )
    # Plaintext fallback shown in notifications and non-Block Kit clients
    fallback = (
        f"Larridin signal: {post['id']} | score={score} | "
        f"categories={','.join(categories)} | {post['url']}"
    )

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {bot_token}"},
            json={"channel": user_id, "text": fallback, "blocks": blocks},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            log.error(f"Slack API error for {post['id']}: {payload.get('error')}")
        else:
            log.info(f"Slack DM sent: {post['id']}")
    except Exception as exc:
        log.error(f"Slack DM failed for {post['id']}: {type(exc).__name__}")


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
    """Score and route posts. Returns (high_count, low_count, discarded_count)."""
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

def run(source: str = "all") -> None:
    global DRY_RUN  # already set by main() before run() is called

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
        description="Larridin social listening agent — monitors Reddit and HN for AI pain signals"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score posts but skip Slack alerts and file writes",
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
        log.info("Dry-run mode enabled — no Slack messages or file writes will occur")

    run(source=args.source)


if __name__ == "__main__":
    main()
