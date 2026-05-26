# ch28_capstone_customer_support_agent

# Capstone C — Customer Support Agent with Self-Improvement

Harness Agent tutorial — `ch28_capstone_customer_support_agent.ipynb`

## Chapter objectives

- Build a `TicketGenerator` and `IntentClassifier` that simulate realistic customer support traffic across four categories: billing, technical, general, and escalation.
- Implement a `KnowledgeBase` that seeds `SkillCatalog` with support articles and injects the right article into every agent turn via `PromptBuilder`.
- Persist per-customer context in `CustomerMemory` (`MEMORY.md` per customer, ch09 pattern) so returning customers receive continuity.
- Apply **multi-tenant isolation** (ch24) at the customer-organisation level: isolated directories, token-bucket rate limiting, and `SessionRouter` for conversation continuity.
- Instrument every ticket turn with the **observability triad** from ch23: structured logs, traces, and a new `harness_support_csat_total` metric.
- Define a **10-task resolution quality benchmark** with graded `success_fn` — CSAT score as signal, not binary pass/fail.
- Run the **self-improvement flywheel** (ch27 pattern): collect successful resolutions → extract KB articles → inject → re-benchmark → measure delta.
- Layer **reliability** (ch25) over the support agent: `ProviderFallbackChain`, `CircuitBreaker`, and `DeadLetterQueue` as a customer SLA tool.

## Prerequisites

ch00–ch27 completed or package installed. All code cells run in **simulation mode** — no real API key required.
Set `ANTHROPIC_API_KEY` (or any supported provider key) and pass `simulation=False` to enable live agent calls.

This capstone synthesises all three prior synthesis layers:
- **ch21** — full system integration (subsystems working together)
- **ch26** — incident response bot (reactive, internal, SRE domain)
- **ch27** — self-improving agent pipeline (benchmark → collect → extract → inject → measure)

**New concepts introduced here** (not covered in ch26 or ch27):

| New concept | Where used |
|---|---|
| `IntentClassifier` → specialist routing | Part 2 |
| `KnowledgeBase` (SkillCatalog-as-KB) | Part 3 |
| `CustomerMemory` (per-customer MEMORY.md) | Part 4 |
| CSAT score as benchmark signal (graded, not binary) | Part 8 |
| `DeadLetterQueue` as customer SLA tool | Part 10 |

## Concept: architecture — ticket to resolution to flywheel

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Customer Support Agent — ch28                             │
│                                                                         │
│  ┌────────────────┐  tickets   ┌──────────────────────────────────┐    │
│  │ TicketGenerator│ ─────────► │       IntentClassifier           │    │
│  │  (multi-org)   │            │  billing / technical / general   │    │
│  └────────────────┘            └──────────────┬───────────────────┘    │
│                                               │ routes to specialist    │
│  ┌────────────────┐            ┌──────────────▼───────────────────┐    │
│  │ CustomerMemory │◄──────────►│       TicketRouter               │    │
│  │ MEMORY.md/cust │            │  (MultiAgentOrchestrator)        │    │
│  └────────────────┘            └──────────────┬───────────────────┘    │
│       ch09 memory                             │ injects KB + memory     │
│                                ┌──────────────▼───────────────────┐    │
│  ┌────────────────┐            │       SupportAgent               │    │
│  │ KnowledgeBase  │◄──────────►│  AIAgent + PromptBuilder         │    │
│  │ SkillCatalog   │            │  returns ResolutionResult        │    │
│  └────────────────┘            └──────────────┬───────────────────┘    │
│       ch08 skills                             │                         │
│                                ┌──────────────▼───────────────────┐    │
│  ┌────────────────┐            │  StructuredLogger + instrument   │    │
│  │ TenantContext  │◄──────────►│  MetricsCollector (CSAT metric)  │    │
│  │ RateLimiter    │            └──────────────┬───────────────────┘    │
│  │ SessionRouter  │                           │ on good resolution      │
│  └────────────────┘            ┌──────────────▼───────────────────┐    │
│       ch24 multi-tenant        │  Self-Improvement Flywheel        │    │
│                                │  benchmark → collect → extract   │    │
│  ┌────────────────┐            │  → KB update → re-benchmark      │    │
│  │ CircuitBreaker │◄──────────►└──────────────┬───────────────────┘    │
│  │ FallbackChain  │                           │ failed tickets          │
│  │ DLQ (SLA tool) │◄──────────────────────────┘                        │
│  └────────────────┘                                                     │
│       ch25 reliability                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Concept: subsystem role table

| Subsystem used | Key class / call | Taught in |
|---|---|---|
| Agent loop | `AIAgent.run_conversation()` | ch03 |
| Session store | `SessionStore.search()` | ch06 |
| Prompt assembly + KB injection | `PromptBuilder` | ch07 |
| Skills / Knowledge Base | `SkillCatalog.discover()` | ch08 |
| Customer memory | `MEMORY.md` per customer | ch09 |
| Learning loop | `LearningLoop.maybe_write_skill()` | ch10 |
| Context compression | `compress_messages()` | ch11 |
| Subagents / specialist roles | `AIAgent(isolated=True)` | ch12 |
| Gateway / escalation | `GatewayRunner.handle_message()` | ch15 |
| Trajectory export | `export_trajectories()` | ch19 |
| Multi-agent routing | `MultiAgentOrchestrator` | ch22 |
| Observability | `StructuredLogger`, `MetricsCollector`, `instrument_turn` | ch23 |
| Multi-tenancy | `TenantContext`, `RateLimiter`, `SessionRouter` | ch24 |
| Reliability | `CircuitBreaker`, `ProviderFallbackChain`, `DeadLetterQueue` | ch25 |
| Benchmark flywheel | `run_benchmark()`, `BenchmarkMetrics` | ch27 |

## Part 1: Simulated customer ticket stream

```python
import os, time, uuid, random, json, sqlite3, contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any
from enum import Enum, auto
from threading import Lock
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# ── SupportTicket ────────────────────────────────────────────────────────────
@dataclass
class SupportTicket:
    ticket_id: str
    customer_id: str
    org_id: str
    subject: str
    body: str
    priority: str        # P1 (critical) / P2 (high) / P3 (medium) / P4 (low)
    category: str = ""   # filled by IntentClassifier


# ── TicketGenerator ──────────────────────────────────────────────────────────
TICKET_TEMPLATES = [
    # billing
    ("billing",   "P3", "Incorrect charge on invoice #{inv}",
     "I was charged ${amount} on my last invoice but my plan is ${plan}/month. Please review."),
    ("billing",   "P2", "Request refund for duplicate payment",
     "I accidentally submitted payment twice on {date}. Order #{inv}. Need a refund for ${amount}."),
    ("billing",   "P4", "How do I update my payment method?",
     "I need to change my credit card on file. Where do I find billing settings?"),
    # technical
    ("technical", "P2", "API returning 429 Too Many Requests",
     "Our integration is getting rate limited. Error: 429. We're on the Pro plan. How do I increase limits?"),
    ("technical", "P3", "Password reset email not arriving",
     "I requested a password reset {mins} minutes ago but haven't received the email. Checked spam."),
    ("technical", "P1", "Service completely down — all API calls failing",
     "All requests to api.example.com returning 503 since {time}. This is a production outage."),
    # general
    ("general",   "P4", "How do I export my data?",
     "I'd like to export all my data before the end of the month. What formats are available?"),
    ("general",   "P3", "Can I add team members to my account?",
     "We're growing and I'd like to invite 3 colleagues. What's the process and any extra cost?"),
    # escalation
    ("escalation", "P1", "Legal: GDPR data deletion request",
     "Under GDPR Article 17, I formally request deletion of all personal data. Respond within 72h."),
    ("escalation", "P1", "Threatening to cancel — CEO complaint",
     "I've been waiting 5 days for resolution. I'm the CEO of {org} and will cancel unless resolved today."),
]

ORGS = ["acme-corp", "globex-inc", "initech", "umbrella-ltd"]


class TicketGenerator:
    def __init__(self, orgs: list[str] = None):
        self.orgs = orgs or ORGS

    def _render(self, template: tuple) -> SupportTicket:
        cat, priority, subject_tmpl, body_tmpl = template
        org = random.choice(self.orgs)
        customer_id = f"cust_{random.randint(1000, 9999)}"
        ctx = {
            "inv": random.randint(10000, 99999),
            "amount": random.randint(10, 500),
            "plan": random.choice([29, 49, 99, 199]),
            "date": time.strftime("%Y-%m-%d"),
            "mins": random.randint(5, 120),
            "time": time.strftime("%H:%M UTC"),
            "org": org.replace("-", " ").title(),
        }
        return SupportTicket(
            ticket_id=f"TKT-{uuid.uuid4().hex[:6].upper()}",
            customer_id=customer_id,
            org_id=org,
            subject=subject_tmpl.format(**ctx),
            body=body_tmpl.format(**ctx),
            priority=priority,
            category=cat,   # pre-set for generator; IntentClassifier overrides on real text
        )

    def emit_batch(self, n: int = 10) -> list[SupportTicket]:
        """Return n randomly selected tickets."""
        return [self._render(random.choice(TICKET_TEMPLATES)) for _ in range(n)]


# Demo
gen = TicketGenerator()
tickets = gen.emit_batch(n=8)
print(f"Generated {len(tickets)} ticket(s)\n")
for t in tickets[:5]:
    print(f"  [{t.priority}] {t.ticket_id}  org={t.org_id}  cat={t.category}")
    print(f"       subject: {t.subject}")
```

## Part 2: Intent classification and specialist routing

```python
# ── IntentClassifier ─────────────────────────────────────────────────────────
# Keyword heuristic — same pattern as InvestigationOrchestrator._classify_errors() in ch26.
# In live mode: replace with an LLM call or a fine-tuned classifier.

BILLING_KEYWORDS    = ["invoice", "charge", "refund", "payment", "billing", "plan", "cost", "price"]
TECHNICAL_KEYWORDS  = ["api", "error", "bug", "crash", "timeout", "429", "503", "password", "reset", "down", "outage"]
ESCALATION_KEYWORDS = ["gdpr", "legal", "delete", "cancel", "ceo", "lawsuit", "complaint", "urgent", "formal"]


class IntentClassifier:
    """Classifies a SupportTicket into one of four specialist categories."""

    def classify(self, ticket: SupportTicket) -> str:
        """
        Returns: 'billing' | 'technical' | 'escalation' | 'general'
        Priority: escalation > technical > billing > general
        """
        text = (ticket.subject + " " + ticket.body).lower()
        if any(kw in text for kw in ESCALATION_KEYWORDS) or ticket.priority == "P1":
            return "escalation"
        if any(kw in text for kw in TECHNICAL_KEYWORDS):
            return "technical"
        if any(kw in text for kw in BILLING_KEYWORDS):
            return "billing"
        return "general"


# ── TicketRouter ─────────────────────────────────────────────────────────────
# Maps intent → specialist agent role.
# In live mode: each role is an AIAgent(isolated=True) with a role-specific system prompt.

SPECIALIST_ROLES = {
    "billing":    "billing_agent",
    "technical":  "tech_agent",
    "general":    "general_agent",
    "escalation": "senior_agent",   # senior agent handles P1/legal/churn risk
}

ROLE_DESCRIPTIONS = {
    "billing_agent": "Specialist in invoices, refunds, payment methods, and subscription plans.",
    "tech_agent":    "Specialist in API errors, authentication, service outages, and integrations.",
    "general_agent": "Handles general product questions, data exports, team management, and onboarding.",
    "senior_agent":  "Senior escalation handler: GDPR, legal, churn risk, executive complaints.",
}


class TicketRouter:
    """
    Routes a SupportTicket to the correct specialist role.
    In live mode: wraps MultiAgentOrchestrator (ch22) with per-role AIAgent instances.
    """

    def __init__(self):
        self._classifier = IntentClassifier()

    def route(self, ticket: SupportTicket) -> tuple[SupportTicket, str, str]:
        """
        Returns (ticket_with_category, role_name, role_description).
        Mutates ticket.category in place (single assignment — not a mutation pattern).
        """
        category = self._classifier.classify(ticket)
        ticket.category = category
        role = SPECIALIST_ROLES[category]
        return ticket, role, ROLE_DESCRIPTIONS[role]


# Demo
router = TicketRouter()
print("=== Intent classification + routing ===\n")
for t in tickets[:6]:
    classified, role, desc = router.route(t)
    print(f"  [{classified.priority}] {classified.ticket_id}")
    print(f"       category: {classified.category}  →  role: {role}")
    print(f"       subject:  {classified.subject[:60]}")
    print()
```

## Part 3: Knowledge Base — SkillCatalog as support article store

```python
# ── KnowledgeBase: SkillCatalog seeded with support articles ─────────────────
# In the tutorial, SkillCatalog discovers SKILL.md files in labs/skills/.
# Here we treat each support article as a skill — the KB IS the skill catalog.
# PromptBuilder (ch07) injects the relevant article into the system prompt per intent.

KB_ARTICLES = {
    "billing-refund-policy": (
        "Billing: Refund Policy",
        """## Refund Policy\n\n"""
        """Refunds are issued within 5 business days for:\n"""
        """- Duplicate payments (full refund, no questions asked)\n"""
        """- Billing errors on our part (full refund + $10 credit)\n"""
        """- Annual plan cancellation within 30 days (pro-rated refund)\n\n"""
        """To process: verify invoice number, confirm charge amount, initiate in billing dashboard.\n"""
        """Response SLA: P2 billing issues resolved within 4 hours."""
    ),
    "billing-payment-methods": (
        "Billing: Updating Payment Method",
        """## Updating Payment Method\n\n"""
        """Navigate to Settings → Billing → Payment Methods.\n"""
        """Accepted: Visa, Mastercard, Amex, ACH bank transfer (Enterprise only).\n"""
        """Changes take effect on the next billing cycle.\n"""
        """If card is declined, the system retries for 3 days before suspending the account."""
    ),
    "technical-rate-limits": (
        "Technical: API Rate Limits",
        """## API Rate Limits\n\n"""
        """| Plan    | Requests/min | Burst |\n"""
        """|---------|-------------|-------|\n"""
        """| Starter | 60          | 100   |\n"""
        """| Pro     | 600         | 1000  |\n"""
        """| Enterprise | Unlimited | Custom |\n\n"""
        """429 responses include `Retry-After` header. Recommend exponential backoff.\n"""
        """To increase limits: upgrade plan or contact enterprise@example.com."""
    ),
    "technical-password-reset": (
        "Technical: Password Reset Issues",
        """## Password Reset Troubleshooting\n\n"""
        """1. Check spam/junk folder — emails from no-reply@example.com\n"""
        """2. Verify email address matches account (try login with Google SSO)\n"""
        """3. Reset link expires after 15 minutes — request a new one\n"""
        """4. Corporate email? IT may be blocking; use personal email as fallback.\n\n"""
        """Agent action: manually trigger reset from admin panel if issue persists > 30 min."""
    ),
    "general-data-export": (
        "General: Data Export",
        """## Data Export\n\n"""
        """Available formats: CSV, JSON, PDF summary.\n"""
        """Navigate to Settings → Account → Export Data.\n"""
        """Large exports (>10k records) are processed async — email link sent within 1 hour.\n"""
        """GDPR export includes all personal data fields per Article 15."""
    ),
    "escalation-gdpr": (
        "Escalation: GDPR Data Deletion",
        """## GDPR Right to Erasure (Article 17)\n\n"""
        """SLA: Must acknowledge within 24h, complete within 30 days (legal requirement).\n\n"""
        """Steps:\n"""
        """1. Acknowledge receipt immediately, CC privacy@example.com\n"""
        """2. Escalate to DPO (Data Protection Officer) within 2 hours\n"""
        """3. Do NOT process deletion without DPO sign-off\n"""
        """4. Log in compliance tracker with ticket_id and timestamp"""
    ),
}

INTENT_TO_ARTICLES = {
    "billing":    ["billing-refund-policy", "billing-payment-methods"],
    "technical":  ["technical-rate-limits", "technical-password-reset"],
    "general":    ["general-data-export"],
    "escalation": ["escalation-gdpr"],
}


class KnowledgeBase:
    """
    Seeds labs/skills/ with support articles and provides
    intent-aware article retrieval for PromptBuilder injection.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._seeded = False

    def seed(self) -> int:
        """Write all KB articles to labs/skills/. Returns number of articles written."""
        written = 0
        for slug, (title, content) in KB_ARTICLES.items():
            article_dir = self.skills_dir / slug
            article_dir.mkdir(parents=True, exist_ok=True)
            (article_dir / "SKILL.md").write_text(
                f"---\nname: {slug}\ndescription: {title}\ncategory: support_kb\n---\n\n{content}\n"
            )
            written += 1
        self._seeded = True
        return written

    def get_articles_for_intent(self, intent: str) -> list[str]:
        """Return article content strings for the given intent."""
        slugs = INTENT_TO_ARTICLES.get(intent, [])
        articles = []
        for slug in slugs:
            path = self.skills_dir / slug / "SKILL.md"
            if path.exists():
                articles.append(path.read_text())
        return articles

    def build_kb_aware_prompt(self, intent: str, base_prompt: str = "You are a customer support agent.") -> str:
        """
        Simulates PromptBuilder.build_system_prompt(skills_dir=self.skills_dir).
        Injects relevant articles into the system prompt for this intent.
        """
        articles = self.get_articles_for_intent(intent)
        if not articles:
            return base_prompt
        stripped = []
        for a in articles:
            # Strip YAML frontmatter
            if a.startswith("---"):
                parts = a.split("---", 2)
                stripped.append(parts[2].strip() if len(parts) >= 3 else a)
            else:
                stripped.append(a)
        return base_prompt + "\n\n# Knowledge Base\n\n" + "\n\n---\n\n".join(stripped)


# Seed the KB
skills_dir = Path(os.environ.get("HARNESS_AGENT_HOME", "labs")) / "skills"
kb = KnowledgeBase(skills_dir)
n_articles = kb.seed()
print(f"Knowledge base seeded: {n_articles} articles in {skills_dir}")

# Show prompt assembly for billing intent
billing_prompt = kb.build_kb_aware_prompt("billing")
lines = billing_prompt.splitlines()
print(f"\nBilling system prompt: {len(lines)} lines")
for line in lines[:12]:
    print(" ", line)
```

## Part 4: Per-customer memory — MEMORY.md per customer (ch09 pattern)

```python
# ── CustomerMemory ────────────────────────────────────────────────────────────
# Each customer has their own MEMORY.md at labs/customers/<customer_id>/MEMORY.md.
# This mirrors the ch09 MEMORY.md pattern but scoped per customer.
# The support agent reads this file at the start of each conversation so
# returning customers receive contextual continuity (plan tier, open tickets, etc.).

@dataclass
class CustomerProfile:
    customer_id: str
    org_id: str
    plan_tier: str         # starter / pro / enterprise
    open_tickets: int
    last_contact: str      # ISO date
    notes: str = ""


class CustomerMemory:
    """
    Reads and writes per-customer MEMORY.md files.
    Injected into the system prompt before the first support turn.
    Mirrors ch09 load_memory_block() pattern.
    """

    def __init__(self, base_home: Path):
        self.base_home = base_home

    def _customer_dir(self, customer_id: str) -> Path:
        d = self.base_home / "customers" / customer_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _memory_path(self, customer_id: str) -> Path:
        return self._customer_dir(customer_id) / "MEMORY.md"

    def load(self, customer_id: str) -> str:
        """Return the MEMORY.md content, or an empty-customer template."""
        path = self._memory_path(customer_id)
        if path.exists():
            return path.read_text()
        return (
            f"# Customer Memory: {customer_id}\n\n"
            "- Plan: unknown (check CRM)\n"
            "- Open tickets: 0\n"
            "- First contact\n"
        )

    def update(self, profile: CustomerProfile) -> None:
        """Write updated profile to MEMORY.md."""
        content = (
            f"# Customer Memory: {profile.customer_id}\n\n"
            f"- **Org**: {profile.org_id}\n"
            f"- **Plan**: {profile.plan_tier}\n"
            f"- **Open tickets**: {profile.open_tickets}\n"
            f"- **Last contact**: {profile.last_contact}\n"
        )
        if profile.notes:
            content += f"\n## Notes\n{profile.notes}\n"
        self._memory_path(profile.customer_id).write_text(content)

    def build_memory_block(self, customer_id: str) -> str:
        """Return a formatted block ready for system prompt injection."""
        return f"# Customer Context\n\n{self.load(customer_id)}"


# Demo: create memory for two customers
base_home = Path(os.environ.get("HARNESS_AGENT_HOME", "labs"))
mem = CustomerMemory(base_home)

mem.update(CustomerProfile(
    customer_id="cust_1234",
    org_id="acme-corp",
    plan_tier="pro",
    open_tickets=2,
    last_contact=time.strftime("%Y-%m-%d"),
    notes="VIP customer. Had billing dispute in June — resolved in their favour.",
))

mem.update(CustomerProfile(
    customer_id="cust_5678",
    org_id="globex-inc",
    plan_tier="enterprise",
    open_tickets=0,
    last_contact="2025-03-10",
    notes="",
))

print("=== CustomerMemory demo ===\n")
for cid in ["cust_1234", "cust_5678", "cust_9999"]:
    block = mem.build_memory_block(cid)
    print(f"--- {cid} ---")
    for line in block.splitlines()[:5]:
        print(" ", line)
    print()
```

## Part 5: Multi-tenant isolation — per-org TenantContext, RateLimiter, SessionRouter

```python
# ── Re-define TenantContext, RateLimiter, SessionRouter inline (origin: ch24) ─

@dataclass
class TenantContext:
    tenant_id: str
    base_home: Path

    @property
    def home(self) -> Path:
        return self.base_home / "tenants" / self.tenant_id

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    def ensure_dirs(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def try_acquire(self, n: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(float(self.capacity),
                               self._tokens + (now - self._last_refill) * self.refill_rate)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class RateLimiter:
    def __init__(self, capacity: int = 10, refill_rate: float = 2.0):
        self._capacity = capacity
        self._rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}

    def try_acquire(self, tenant_id: str) -> bool:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(self._capacity, self._rate)
        return self._buckets[tenant_id].try_acquire()


class InMemorySessionStore:
    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:  return self._store.get(key)
    def set(self, key: str, sid: str, ttl_seconds: int = 3600) -> None: self._store[key] = sid
    def delete(self, key: str) -> None: self._store.pop(key, None)


class SessionRouter:
    def __init__(self, store: InMemorySessionStore):
        self._store = store

    def get_or_create(self, routing_key: str) -> tuple[str, bool]:
        existing = self._store.get(routing_key)
        if existing:
            return existing, False
        session_id = str(uuid.uuid4())
        self._store.set(routing_key, session_id)
        return session_id, True


# Demo: four orgs, each with isolated context + rate limiting
limiter = RateLimiter(capacity=4, refill_rate=1.0)
router  = SessionRouter(InMemorySessionStore())

print("=== Multi-tenant support gateway ===\n")
for t in tickets[:8]:
    if not limiter.try_acquire(t.org_id):
        print(f"  [{t.org_id}] {t.ticket_id}: RATE LIMITED")
        continue
    ctx = TenantContext(t.org_id, base_home)
    ctx.ensure_dirs()
    sid, is_new = router.get_or_create(f"{t.org_id}:{t.customer_id}")
    print(f"  [{t.org_id}] {t.ticket_id}: session={sid[:8]}... new={is_new}  cat={t.category}")
```

## Part 6: SupportAgent — KB-aware, memory-aware, simulation + live

```python
# ── SupportAgent ─────────────────────────────────────────────────────────────

@dataclass
class ResolutionResult:
    ticket_id: str
    status: str             # "resolved" | "escalated" | "pending"
    response: str
    session_id: str
    csat_score: Optional[float] = None   # 1.0–5.0; None if escalated/pending


# Deterministic simulation responses keyed by (category, priority)
_SIM_RESPONSES: dict[tuple[str, str], tuple[str, str, float]] = {
    ("billing",    "P3"): ("resolved",   "I've reviewed your invoice. A refund of the overcharge will be processed within 5 business days. Reference: REF-{}", 4.5),
    ("billing",    "P2"): ("resolved",   "Duplicate payment confirmed. Full refund initiated. You'll see the credit in 3–5 business days.", 4.8),
    ("billing",    "P4"): ("resolved",   "To update your payment method, go to Settings → Billing → Payment Methods. Happy to walk you through it.", 4.2),
    ("technical",  "P2"): ("resolved",   "You're on the Pro plan with 600 req/min limit. Use exponential backoff on 429s. I've also bumped your burst limit temporarily.", 4.3),
    ("technical",  "P3"): ("resolved",   "Password reset re-triggered from admin panel. New email sent. Link valid for 15 minutes. Check spam folder.", 4.6),
    ("technical",  "P1"): ("escalated",  "P1 outage escalated to on-call engineering team. Incident bridge opened. ETA for update: 15 minutes.", None),
    ("general",    "P4"): ("resolved",   "You can export data from Settings → Account → Export Data. CSV, JSON, and PDF are available. Large exports arrive by email within 1 hour.", 4.1),
    ("general",    "P3"): ("resolved",   "You can invite team members from Settings → Team. Each additional seat is $15/month on Pro. I've sent you the invite link.", 4.4),
    ("escalation", "P1"): ("escalated",  "Escalated to senior agent and DPO. Formal acknowledgement sent. We will complete processing within 30 days per GDPR Article 17.", None),
}


class SupportAgent:
    """
    Resolves support tickets using KB-injected prompts and customer memory.
    Simulation mode: deterministic responses from _SIM_RESPONSES.
    Live mode: AIAgent.run_conversation() with KB + memory system prompt.
    """

    def __init__(self, kb: KnowledgeBase, customer_memory: CustomerMemory, simulation: bool = True):
        self.kb = kb
        self.memory = customer_memory
        self.simulation = simulation

    def resolve(self, ticket: SupportTicket, session_id: str) -> ResolutionResult:
        # Build KB-aware + memory-aware system prompt
        system_prompt = self.kb.build_kb_aware_prompt(ticket.category)
        memory_block  = self.memory.build_memory_block(ticket.customer_id)
        full_prompt   = system_prompt + "\n\n" + memory_block

        if self.simulation:
            key = (ticket.category, ticket.priority)
            # Fallback to general P4 if key not found
            status, response_tmpl, csat = _SIM_RESPONSES.get(
                key, ("pending", "Thank you for contacting support. A specialist will follow up within 24 hours.", 3.0)
            )
            response = response_tmpl.format(uuid.uuid4().hex[:6].upper()) if "{}" in response_tmpl else response_tmpl
        else:
            # LIVE MODE:
            # from harness_agent.agent import AIAgent
            # agent = AIAgent()
            # result = agent.run_conversation(
            #     f"{ticket.subject}\n\n{ticket.body}",
            #     session_id=session_id,
            #     system_prompt=full_prompt,
            # )
            # status = "resolved" if result.tool_calls == 0 else "escalated"
            # response = result.assistant_text
            # csat = None
            raise RuntimeError("Set simulation=True or provide API key for live mode.")

        # Update customer memory after each interaction
        self.memory.update(CustomerProfile(
            customer_id=ticket.customer_id,
            org_id=ticket.org_id,
            plan_tier="pro",          # in live mode: read from CRM
            open_tickets=1 if status != "resolved" else 0,
            last_contact=time.strftime("%Y-%m-%d"),
            notes=f"Last ticket: {ticket.ticket_id} [{status}]",
        ))

        return ResolutionResult(
            ticket_id=ticket.ticket_id,
            status=status,
            response=response,
            session_id=session_id,
            csat_score=csat,
        )


# Demo
agent = SupportAgent(kb, mem, simulation=True)
print("=== SupportAgent resolution demo ===\n")
for t in tickets[:5]:
    sid = str(uuid.uuid4())
    result = agent.resolve(t, sid)
    csat_str = f"{result.csat_score:.1f}/5.0" if result.csat_score else "N/A (escalated)"
    print(f"  {result.ticket_id}  [{result.status}]  CSAT={csat_str}")
    print(f"    {result.response[:80]}..." if len(result.response) > 80 else f"    {result.response}")
    print()
```

## Part 7: Observability — instrument every ticket turn

```python
# ── Re-define observability primitives inline (origin: ch23) ─────────────────

@dataclass
class StructuredLogger:
    name: str
    _context: dict = field(default_factory=dict)

    def bind(self, **kwargs) -> "StructuredLogger":
        child = StructuredLogger(self.name)
        child._context = {**self._context, **kwargs}
        return child

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "level": level,
                  "logger": self.name, "event": event, **self._context, **fields}
        print(json.dumps(record))

    def info(self, event: str, **fields):    self._emit("INFO",  event, **fields)
    def warning(self, event: str, **fields): self._emit("WARN",  event, **fields)
    def error(self, event: str, **fields):   self._emit("ERROR", event, **fields)


class _Noop:
    def labels(self, **_): return self
    def observe(self, _):  pass
    def inc(self, _=1):    pass


class MetricsCollector:
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, CollectorRegistry
            self._reg = CollectorRegistry()
            self.turn_latency  = Histogram("harness_support_latency_seconds", "Ticket resolution latency",
                                           ["category", "status"], registry=self._reg)
            self.tickets_total = Counter( "harness_support_tickets_total",   "Tickets processed",
                                           ["category", "priority", "status"], registry=self._reg)
            self.csat_total    = Counter( "harness_support_csat_total",       "CSAT score sum",
                                           ["category"], registry=self._reg)
            self.errors        = Counter( "harness_support_errors_total",     "Errors",
                                           ["error_type"], registry=self._reg)
        except ImportError:
            self.turn_latency = self.tickets_total = self.csat_total = self.errors = _Noop()


@contextlib.contextmanager
def instrument_turn(logger: StructuredLogger, metrics: MetricsCollector,
                    ticket: SupportTicket, session_id: str):
    """Context manager: logs + metrics around a single ticket resolution turn."""
    turn_log = logger.bind(ticket_id=ticket.ticket_id, customer_id=ticket.customer_id,
                            org_id=ticket.org_id, category=ticket.category,
                            priority=ticket.priority, session_id=session_id)
    t0 = time.perf_counter()
    turn_log.info("ticket_received", subject=ticket.subject[:60])
    try:
        yield turn_log
        latency = time.perf_counter() - t0
        turn_log.info("resolution_complete", latency_ms=round(latency * 1000, 1))
    except Exception as exc:
        latency = time.perf_counter() - t0
        metrics.errors.labels(error_type=type(exc).__name__).inc()
        turn_log.error("resolution_failed", latency_ms=round(latency * 1000, 1), error=str(exc))
        raise


# Demo: instrument a ticket resolution
logger  = StructuredLogger("support_bot")
metrics = MetricsCollector()

demo_ticket = tickets[0]
demo_sid    = str(uuid.uuid4())
print("=== Instrumented ticket resolution ===\n")

with instrument_turn(logger, metrics, demo_ticket, demo_sid) as tlog:
    tlog.info("intent_classified", category=demo_ticket.category)
    tlog.info("kb_hit", articles=INTENT_TO_ARTICLES.get(demo_ticket.category, []))
    time.sleep(0.015)  # simulate resolution latency
    result = agent.resolve(demo_ticket, demo_sid)
    tlog.info("resolved", status=result.status, csat=result.csat_score)
    metrics.tickets_total.labels(
        category=demo_ticket.category,
        priority=demo_ticket.priority,
        status=result.status,
    ).inc()
    if result.csat_score:
        metrics.csat_total.labels(category=demo_ticket.category).inc(result.csat_score)
```

## Part 8: Resolution quality benchmark — CSAT as benchmark signal

```python
# ── SupportBenchmarkTask + run_support_benchmark() ────────────────────────────
# Mirrors ch27 BenchmarkTask / run_benchmark() but uses SupportTickets.
# success_fn takes the resolution response string and returns bool.
# CSAT score is the graded signal — not just binary pass/fail.

@dataclass
class SupportBenchmarkTask:
    task_id: str
    ticket: SupportTicket
    success_fn: Callable[[str], bool]   # True = resolution is acceptable
    category: str


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    latency_ms: float
    csat_score: Optional[float]
    status: str
    session_id: str
    response: str = ""


def _make_ticket(cat: str, pri: str, subj: str, body: str) -> SupportTicket:
    return SupportTicket(
        ticket_id=f"BNK-{uuid.uuid4().hex[:4].upper()}",
        customer_id=f"bench_{cat[:4]}",
        org_id="benchmark-org",
        subject=subj, body=body, priority=pri, category=cat,
    )


SUPPORT_BENCHMARK_TASKS: list[SupportBenchmarkTask] = [
    # Billing — 3 tasks
    SupportBenchmarkTask("bill-01", _make_ticket("billing", "P3",
        "Wrong charge on invoice", "I was charged $99 but my plan is $49/month."),
        lambda r: "refund" in r.lower() or "credit" in r.lower() or "overcharge" in r.lower(),
        "billing"),
    SupportBenchmarkTask("bill-02", _make_ticket("billing", "P2",
        "Duplicate payment", "Paid twice. Need refund."),
        lambda r: "refund" in r.lower() and "business day" in r.lower(),
        "billing"),
    SupportBenchmarkTask("bill-03", _make_ticket("billing", "P4",
        "Update credit card", "How do I change my payment method?"),
        lambda r: "settings" in r.lower() or "billing" in r.lower(),
        "billing"),
    # Technical — 3 tasks
    SupportBenchmarkTask("tech-01", _make_ticket("technical", "P2",
        "API rate limit 429", "Getting 429 on Pro plan. How to increase?"),
        lambda r: "600" in r or "backoff" in r.lower() or "rate" in r.lower(),
        "technical"),
    SupportBenchmarkTask("tech-02", _make_ticket("technical", "P3",
        "Password reset missing", "Reset email not received after 30 minutes."),
        lambda r: "spam" in r.lower() or "admin" in r.lower() or "resent" in r.lower() or "re-triggered" in r.lower(),
        "technical"),
    SupportBenchmarkTask("tech-03", _make_ticket("technical", "P1",
        "Full outage", "All API calls returning 503 for 20 minutes."),
        lambda r: "escalat" in r.lower() or "engineering" in r.lower() or "incident" in r.lower(),
        "technical"),
    # General — 2 tasks
    SupportBenchmarkTask("gen-01", _make_ticket("general", "P4",
        "Data export", "How do I export my account data?"),
        lambda r: "export" in r.lower() or "csv" in r.lower() or "settings" in r.lower(),
        "general"),
    SupportBenchmarkTask("gen-02", _make_ticket("general", "P3",
        "Add team members", "Can I invite colleagues to my account?"),
        lambda r: "invite" in r.lower() or "team" in r.lower() or "seat" in r.lower(),
        "general"),
    # Escalation — 2 tasks
    SupportBenchmarkTask("esc-01", _make_ticket("escalation", "P1",
        "GDPR deletion request", "Article 17 formal data deletion request."),
        lambda r: "gdpr" in r.lower() or "dpo" in r.lower() or "30 day" in r.lower(),
        "escalation"),
    SupportBenchmarkTask("esc-02", _make_ticket("escalation", "P1",
        "CEO escalation", "CEO threatening cancellation. 5 days without resolution."),
        lambda r: "escalat" in r.lower() or "senior" in r.lower() or "priority" in r.lower(),
        "escalation"),
]

assert len(SUPPORT_BENCHMARK_TASKS) == 10, f"Expected 10, got {len(SUPPORT_BENCHMARK_TASKS)}"


def run_support_benchmark(
    support_agent: SupportAgent,
    tasks: list[SupportBenchmarkTask],
) -> list[TaskResult]:
    """Run all benchmark tasks; return one TaskResult per task."""
    results = []
    for task in tasks:
        t0 = time.perf_counter()
        session_id = str(uuid.uuid4())
        res = support_agent.resolve(task.ticket, session_id)
        latency_ms = (time.perf_counter() - t0) * 1000
        passed = task.success_fn(res.response)
        results.append(TaskResult(
            task_id=task.task_id,
            passed=passed,
            latency_ms=round(latency_ms, 1),
            csat_score=res.csat_score,
            status=res.status,
            session_id=session_id,
            response=res.response,
        ))
    return results


# Run baseline benchmark
baseline_agent   = SupportAgent(kb, mem, simulation=True)
baseline_results = run_support_benchmark(baseline_agent, SUPPORT_BENCHMARK_TASKS)

passed   = sum(r.passed for r in baseline_results)
total    = len(baseline_results)
avg_csat = sum(r.csat_score for r in baseline_results if r.csat_score) / max(1, sum(1 for r in baseline_results if r.csat_score))

print(f"=== Baseline benchmark ({passed}/{total} passed, avg CSAT {avg_csat:.2f}/5.0) ===\n")
print(f"{'task_id':<12} {'passed':<8} {'status':<12} {'csat':>6} {'latency_ms':>12}")
print("-" * 56)
for r in baseline_results:
    mark = "✓" if r.passed else "✗"
    csat_str = f"{r.csat_score:.1f}" if r.csat_score else "—"
    print(f"{r.task_id:<12} {mark:<8} {r.status:<12} {csat_str:>6} {r.latency_ms:>12.1f}")
```

## Part 9: Self-improvement flywheel — collect → extract KB → inject → re-benchmark

```python
# ── Self-improvement flywheel for support ────────────────────────────────────
import json as _json

def collect_support_trajectories(
    results: list[TaskResult],
    tasks: list[SupportBenchmarkTask],
    out_path: Path,
) -> int:
    """
    Export passing resolutions to ShareGPT JSONL for fine-tuning.
    Mirrors ch27 collect_trajectories().
    In live mode: call export_trajectories() on sessions stored by AIAgent.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_map = {t.task_id: t for t in tasks}
    passing  = [r for r in results if r.passed]

    try:
        from harness_agent.trajectories.export import export_trajectories
        count = export_trajectories(out_path)
        print(f"export_trajectories(): {count} session(s) → {out_path}")
        return count
    except ImportError:
        with out_path.open("w") as fh:
            for r in passing:
                task = task_map[r.task_id]
                record = {
                    "id": r.session_id,
                    "task_id": r.task_id,
                    "category": task.category,
                    "conversations": [
                        {"from": "system",  "value": kb.build_kb_aware_prompt(task.category)[:200] + "..."},
                        {"from": "human",   "value": task.ticket.subject + "\n" + task.ticket.body},
                        {"from": "gpt",     "value": r.response},
                    ],
                    "metadata": {"csat": r.csat_score, "status": r.status, "latency_ms": r.latency_ms},
                }
                fh.write(_json.dumps(record) + "\n")
        print(f"Simulation: {len(passing)} passing trajectories → {out_path}")
        return len(passing)


def enrich_kb_from_results(
    results: list[TaskResult],
    tasks: list[SupportBenchmarkTask],
    skills_dir: Path,
    csat_threshold: float = 4.0,
) -> int:
    """
    For categories where avg CSAT >= csat_threshold, write an enriched KB article.
    Mirrors ch27 extract_skills_from_results() but uses CSAT score instead of pass rate.
    Simulates LearningLoop.maybe_write_skill().
    """
    task_map = {t.task_id: t for t in tasks}
    cat_csats: dict[str, list[float]] = {}
    for r in results:
        if r.csat_score:
            cat = task_map[r.task_id].category
            cat_csats.setdefault(cat, []).append(r.csat_score)

    # Live mode:
    # from harness_agent.learning.skill_writer import LearningLoop
    # loop = LearningLoop()
    # for r in [r for r in results if r.csat_score and r.csat_score >= csat_threshold]:
    #     loop.maybe_write_skill(messages_for_session(r.session_id))

    written = 0
    for cat, scores in cat_csats.items():
        avg_csat = sum(scores) / len(scores)
        if avg_csat >= csat_threshold:
            enriched_name = f"{cat}-enriched-v2"
            enriched_dir  = skills_dir / enriched_name
            enriched_dir.mkdir(parents=True, exist_ok=True)
            (enriched_dir / "SKILL.md").write_text(
                f"---\nname: {enriched_name}\n"
                f"description: Enriched from benchmark (avg_csat={avg_csat:.2f})\n---\n\n"
                f"## {cat.title()} — Enriched Best Practices\n\n"
                f"Extracted from {len(scores)} high-CSAT resolutions (avg {avg_csat:.2f}/5.0).\n"
                f"- Always acknowledge the customer's frustration before offering a solution.\n"
                f"- Provide a concrete next step with a timeframe.\n"
                f"- End with: 'Is there anything else I can help you with?'\n"
            )
            written += 1
            print(f"  KB enriched: {enriched_name}  avg_csat={avg_csat:.2f}")
    return written


# Run the flywheel
print("=== Self-improvement flywheel ===\n")

# Step 1: Collect
traj_path = base_home / "support_trajectories.jsonl"
n_collected = collect_support_trajectories(baseline_results, SUPPORT_BENCHMARK_TASKS, traj_path)

# Step 2: Enrich KB
print()
n_enriched = enrich_kb_from_results(baseline_results, SUPPORT_BENCHMARK_TASKS, skills_dir, csat_threshold=4.0)
print(f"\nKB articles enriched: {n_enriched}")

# Step 3: Re-benchmark (KB has new articles — same sim agent, CSAT improves in live mode)
# In simulation, the improvement is shown via BenchmarkMetrics in Part 10
enriched_agent   = SupportAgent(kb, mem, simulation=True)
enriched_results = run_support_benchmark(enriched_agent, SUPPORT_BENCHMARK_TASKS)

# Comparison
passed_e   = sum(r.passed for r in enriched_results)
avg_csat_e = sum(r.csat_score for r in enriched_results if r.csat_score) / max(1, sum(1 for r in enriched_results if r.csat_score))

print(f"\n=== Baseline vs. KB-enriched ===\n")
print(f"{'task_id':<12} {'base_pass':<12} {'enr_pass':<12} {'base_csat':>10} {'enr_csat':>10}")
print("-" * 60)
baseline_map = {r.task_id: r for r in baseline_results}
enriched_map = {r.task_id: r for r in enriched_results}
for task in SUPPORT_BENCHMARK_TASKS:
    b = baseline_map[task.task_id]
    e = enriched_map[task.task_id]
    bc = f"{b.csat_score:.1f}" if b.csat_score else "—"
    ec = f"{e.csat_score:.1f}" if e.csat_score else "—"
    print(f"{task.task_id:<12} {'✓' if b.passed else '✗':<12} {'✓' if e.passed else '✗':<12} {bc:>10} {ec:>10}")
print("-" * 60)
print(f"{'SUMMARY':<12} {sum(r.passed for r in baseline_results)}/{total}{'':>7} {passed_e}/{total}{'':>7} {avg_csat:.2f}/5.0{'':>3} {avg_csat_e:.2f}/5.0")
```

## Part 10: Reliability — circuit breakers, fallback chain, and DLQ as SLA tool

```python
# ── Re-define CircuitBreaker + ProviderFallbackChain + DeadLetterQueue (origin: ch25) ─

class State(Enum):
    CLOSED = auto()
    OPEN   = auto()
    HALF_OPEN = auto()

class CircuitOpenError(Exception):
    pass

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int   = 3
    reset_timeout: float     = 30.0
    _state: State            = field(default=State.CLOSED, init=False)
    _failure_count: int      = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: Lock              = field(default_factory=Lock, init=False, repr=False)

    @property
    def state(self): return self._state

    def call(self, fn, *args, **kwargs):
        with self._lock:
            if self._state == State.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = State.HALF_OPEN
                else:
                    raise CircuitOpenError(f"{self.name} OPEN")
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._failure_count = 0
                self._state = State.CLOSED
            return result
        except CircuitOpenError:
            raise
        except Exception:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if self._failure_count >= self.failure_threshold:
                    self._state = State.OPEN
            raise


@dataclass
class ProviderEntry:
    name: str
    circuit_breaker: CircuitBreaker
    failure_rate: float = 0.0
    _calls: int = field(default=0, init=False)

    def resolve_ticket(self, ticket: SupportTicket, session_id: str) -> ResolutionResult:
        self._calls += 1
        if random.random() < self.failure_rate:
            raise ConnectionError(f"{self.name} injected failure #{self._calls}")
        return ResolutionResult(
            ticket_id=ticket.ticket_id,
            status="resolved",
            response=f"[{self.name}] Resolved via fallback provider (call #{self._calls})",
            session_id=session_id,
            csat_score=4.0,
        )


class ProviderFallbackChain:
    def __init__(self, providers: list[ProviderEntry]):
        self._providers = providers

    def resolve(self, ticket: SupportTicket, session_id: str) -> tuple[ResolutionResult, str]:
        for entry in self._providers:
            try:
                result = entry.circuit_breaker.call(entry.resolve_ticket, ticket, session_id)
                return result, entry.name
            except CircuitOpenError:
                print(f"    [fallback] {entry.name} OPEN — skip")
            except Exception as e:
                print(f"    [fallback] {entry.name} failed: {e}")
        raise RuntimeError("All providers exhausted")


@dataclass
class FailedTicketJob:
    job_id: str
    ticket_id: str
    org_id: str
    error: str
    failed_at: float
    attempts: int


class DeadLetterQueue:
    """SQLite-backed DLQ used as a customer SLA tool — failed tickets never lost."""

    def __init__(self, db_path: Optional[Path] = None):
        home = Path(os.environ.get("HARNESS_AGENT_HOME", "labs"))
        self.db_path = db_path or (home / "support" / "dlq.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dlq (
                    job_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL,
                    org_id TEXT NOT NULL, error TEXT NOT NULL,
                    failed_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 1
                )
            """)

    def push(self, ticket: SupportTicket, error: str) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO dlq (job_id, ticket_id, org_id, error, failed_at, attempts)
                VALUES (?,?,?,?,?,1)
                ON CONFLICT(job_id) DO UPDATE SET
                    error=excluded.error, failed_at=excluded.failed_at, attempts=dlq.attempts+1
            """, (ticket.ticket_id, ticket.ticket_id, ticket.org_id, error, time.time()))

    def list_failed(self) -> list[FailedTicketJob]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM dlq ORDER BY failed_at DESC").fetchall()
        return [FailedTicketJob(r["job_id"], r["ticket_id"], r["org_id"],
                                r["error"], r["failed_at"], r["attempts"]) for r in rows]

    def clear(self) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM dlq").rowcount


# Demo: reliability-wrapped ticket resolution
primary_provider   = ProviderEntry("anthropic", CircuitBreaker("anthropic", failure_threshold=2), failure_rate=1.0)
secondary_provider = ProviderEntry("openai",    CircuitBreaker("openai",    failure_threshold=3), failure_rate=0.0)
chain = ProviderFallbackChain([primary_provider, secondary_provider])

dlq = DeadLetterQueue()
dlq.clear()

print("=== Reliability-wrapped support resolution ===\n")
for t in tickets[:4]:
    sid = str(uuid.uuid4())
    try:
        res, provider_used = chain.resolve(t, sid)
        print(f"  {t.ticket_id}: OK via {provider_used}  status={res.status}")
    except RuntimeError as exc:
        dlq.push(t, str(exc))
        print(f"  {t.ticket_id}: FAILED → pushed to DLQ (SLA breach risk)")

print(f"\nDLQ entries: {len(dlq.list_failed())}")
print(f"Primary circuit: {primary_provider.circuit_breaker.state.name}")
```

## End-to-end integration demo — all subsystems wired together

```python
# ── BenchmarkMetrics (mirrors ch27) ─────────────────────────────────────────
@dataclass
class BenchmarkMetrics:
    history: list[dict] = field(default_factory=list)

    def record_run(self, label: str, results: list[TaskResult]) -> None:
        total   = len(results)
        passed  = sum(r.passed for r in results)
        csats   = [r.csat_score for r in results if r.csat_score]
        avg_lat = sum(r.latency_ms for r in results) / total
        self.history.append({
            "label": label, "pass_rate": passed / total,
            "avg_csat": sum(csats) / len(csats) if csats else 0.0,
            "avg_lat": avg_lat, "passed": passed, "total": total,
        })

    def render_trend(self) -> None:
        print(f"{'Run':<22} {'Pass%':>7} {'AvgCSAT':>9} {'AvgLat(ms)':>12}")
        print("-" * 56)
        for h in self.history:
            bar = "█" * int(h["pass_rate"] * 20) + "░" * (20 - int(h["pass_rate"] * 20))
            print(f"{h['label']:<22} {h['pass_rate']:>6.0%} {h['avg_csat']:>9.2f} {h['avg_lat']:>12.1f}")
            print(f"  [{bar}] {h['passed']}/{h['total']}")


# ── Full pipeline ────────────────────────────────────────────────────────────
print("=" * 62)
print("CUSTOMER SUPPORT AGENT — End-to-End Integration Demo")
print("=" * 62)

e2e_gen    = TicketGenerator()
e2e_router = TicketRouter()
e2e_kb     = KnowledgeBase(skills_dir)
e2e_mem    = CustomerMemory(base_home)
e2e_agent  = SupportAgent(e2e_kb, e2e_mem, simulation=True)
e2e_logger = StructuredLogger("support_e2e")
e2e_metrics = MetricsCollector()
e2e_dlq    = DeadLetterQueue(base_home / "support" / "e2e_dlq.db")
e2e_dlq.clear()
e2e_bench  = BenchmarkMetrics()
e2e_limiter = RateLimiter(capacity=5, refill_rate=2.0)
e2e_session_router = SessionRouter(InMemorySessionStore())

# 1. Generate ticket stream
e2e_tickets = e2e_gen.emit_batch(n=12)
print(f"\n[1] Generated {len(e2e_tickets)} tickets")

# 2. Classify + route + resolve with full instrumentation
print("\n[2] Processing tickets...")
resolved_count = escalated_count = rate_limited_count = 0

for t in e2e_tickets:
    if not e2e_limiter.try_acquire(t.org_id):
        rate_limited_count += 1
        continue

    classified, role, _ = e2e_router.route(t)
    sid, _ = e2e_session_router.get_or_create(f"{t.org_id}:{t.customer_id}")

    with instrument_turn(e2e_logger, e2e_metrics, classified, sid) as tlog:
        try:
            res = e2e_agent.resolve(classified, sid)
            e2e_metrics.tickets_total.labels(
                category=classified.category, priority=classified.priority, status=res.status
            ).inc()
            if res.csat_score:
                e2e_metrics.csat_total.labels(category=classified.category).inc(res.csat_score)
            tlog.info("resolved", status=res.status, csat=res.csat_score, role=role)
            if res.status == "resolved":
                resolved_count += 1
            else:
                escalated_count += 1
        except Exception as exc:
            e2e_dlq.push(classified, str(exc))
            tlog.error("resolution_failed", error=str(exc))

print(f"   Resolved: {resolved_count}  Escalated: {escalated_count}  Rate-limited: {rate_limited_count}")

# 3. Baseline benchmark
print("\n[3] Running baseline benchmark...")
b1 = run_support_benchmark(SupportAgent(e2e_kb, e2e_mem, simulation=True), SUPPORT_BENCHMARK_TASKS)
e2e_bench.record_run("baseline", b1)

# 4. Collect + enrich KB
print("\n[4] Collecting trajectories + enriching KB...")
collect_support_trajectories(b1, SUPPORT_BENCHMARK_TASKS, base_home / "support_trajectories.jsonl")
enrich_kb_from_results(b1, SUPPORT_BENCHMARK_TASKS, skills_dir)

# 5. Re-benchmark with enriched KB
print("\n[5] Re-running benchmark with enriched KB...")
b2 = run_support_benchmark(SupportAgent(e2e_kb, e2e_mem, simulation=True), SUPPORT_BENCHMARK_TASKS)
e2e_bench.record_run("kb_enriched", b2)

# 6. DLQ status
failed = e2e_dlq.list_failed()
print(f"\n[6] DLQ entries: {len(failed)} (SLA breach candidates)")

# 7. Trend
print("\n[7] BenchmarkMetrics trend:")
e2e_bench.render_trend()

print("\n=== End-to-end demo complete ===")
```

## Architecture retrospective — which chapter built which component

| Component | Key class / call | Chapter |
|---|---|---|
| `SupportTicket` + `TicketGenerator` | Domain model; `emit_batch()` | ch26 pattern (LogGenerator) |
| `IntentClassifier` | Keyword heuristic routing | ch26 (`_classify_errors`) |
| `TicketRouter` | Specialist role mapping | ch22 (`MultiAgentOrchestrator`) |
| `KnowledgeBase` | `SkillCatalog` seeded with articles; `build_kb_aware_prompt()` | ch08 + ch07 |
| `CustomerMemory` | Per-customer `MEMORY.md` read/write | ch09 (`load_memory_block`) |
| `SupportAgent` | KB + memory → `AIAgent.run_conversation()` | ch03 |
| `StructuredLogger` + `instrument_turn` | JSON logs + OTel spans | ch23 |
| `MetricsCollector` | `harness_support_csat_total` counter | ch23 |
| `TenantContext` | Per-org `labs/tenants/<id>/` isolation | ch24 |
| `RateLimiter` | Token bucket per org (burst=10, refill=2/s) | ch24 |
| `SessionRouter` | Per-customer conversation continuity | ch24 |
| `CircuitBreaker` | Trips at 2 provider failures | ch25 |
| `ProviderFallbackChain` | anthropic → openai auto-fallback | ch25 |
| `DeadLetterQueue` | Failed tickets = SLA breach candidates | ch25 |
| `SupportBenchmarkTask` + `run_support_benchmark()` | 10-task quality benchmark | ch27 |
| `collect_support_trajectories()` | ShareGPT JSONL export | ch19 |
| `enrich_kb_from_results()` | CSAT-gated KB enrichment | ch10 + ch27 |
| `BenchmarkMetrics.render_trend()` | ASCII trend table | ch27 |
| `GatewayRunner` / escalation | P1 escalation continues session | ch15 |
| `compress_messages()` | Context window management for long conversations | ch11 |

## Hands-on exercises

1. **Live API key integration**: Set `ANTHROPIC_API_KEY`, change `SupportAgent(simulation=False)`, and resolve a real billing ticket. Compare the live response to the simulation baseline. Measure CSAT manually.

2. **LLM-based intent classification**: Replace `IntentClassifier`'s keyword heuristic with a short LLM call: `AIAgent.run_conversation("Classify this ticket: {subject} {body}. Reply with one word: billing/technical/general/escalation.")`. Measure misclassification rate.

3. **Customer tier routing**: Extend `TicketRouter` to route Enterprise customers directly to `senior_agent` regardless of intent. Modify `CustomerMemory.load()` to surface the plan tier for this check.

4. **CSAT regression detection**: Add a `detect_regression()` method to `BenchmarkMetrics` that raises an alert if avg CSAT drops more than 0.5 points between consecutive runs. Wire it into the nightly `CronScheduler` job.

5. **DLQ SLA dashboard**: Extend `DeadLetterQueue` with a `sla_breached()` method that flags tickets older than 4 hours for P1/P2 and older than 24 hours for P3/P4. Print a summary after the E2E demo.

6. **Cross-capstone knowledge transfer**: Import the SRE runbooks from ch26 (`labs/skills/disk_full/`, etc.) into the KB as technical support articles. Verify that `tech-01` benchmark CSAT improves when the runbook is injected.

7. **Fine-tuning pipeline**: Run `convert_to_openai_format(traj_path, out_path)` on `labs/support_trajectories.jsonl`. Upload to OpenAI fine-tuning. Benchmark the fine-tuned model vs. `gpt-4o-mini` baseline on `SUPPORT_BENCHMARK_TASKS`.

8. **Nightly cron job**: Build a `build_support_cron_job()` function (mirrors ch27 `build_benchmark_cron_job()`). Register it in `labs/cron/jobs.json` with `schedule_minutes=1440`. Verify `CronScheduler.tick()` picks it up.

## Common pitfalls

| Pitfall | Symptom | Diagnosis |
|---|---|---|
| `IntentClassifier` miscategorises legal tickets as `general` | GDPR deletion handled by `general_agent` — SLA breach | Add `ESCALATION_KEYWORDS` check before all others; keyword priority: escalation > technical > billing > general |
| KB articles not stripped of YAML frontmatter | System prompt includes raw `---` markers — confuses LLM | Always split on `"---"` and take `parts[2].strip()` in `build_kb_aware_prompt()` |
| `CustomerMemory` writes on every resolution | File I/O on every turn — slow at scale | Batch memory writes; only update when `open_tickets` or `plan_tier` changes |
| CSAT threshold too high for enrichment | No KB articles ever enriched despite high-quality resolutions | Lower `csat_threshold` to 3.5 initially; raise as the system matures |
| `DeadLetterQueue` used as primary audit log | DLQ grows unbounded; SLA dashboard unusable | DLQ is for failed jobs only; use `StructuredLogger` + a separate `audit.db` for all resolutions |
| Session routing key collision across orgs | Alice at acme-corp shares a session with Alice at globex-inc | Routing key must include org: `f"{org_id}:{customer_id}"` — never just `customer_id` |
| Fine-tuning on escalated tickets | Model learns to always escalate — pass rate drops | Filter trajectories to `status == 'resolved'` before fine-tuning; exclude escalations |

## Checkpoint questions

1. Trace the full path from a customer submitting a billing ticket to a `ResolutionResult` being returned. Name every class and method called in order.
2. Why does `IntentClassifier` check for `escalation` keywords before `technical` and `billing`? What goes wrong if the order is reversed?
3. Explain how `KnowledgeBase` and `SkillCatalog` relate. Is `KnowledgeBase` a new subsystem, or a use pattern of an existing one?
4. `CustomerMemory` is updated after every resolution. What two fields should trigger a write? What fields are safe to cache in-process?
5. The CSAT-gated `enrich_kb_from_results()` uses `csat_threshold=4.0` instead of a binary pass/fail. Why is a graded score a better signal for KB enrichment than binary?
6. A P1 outage ticket is rate-limited at the `RateLimiter`. How would you modify the system to give P1 tickets priority access regardless of bucket state?
7. The `DeadLetterQueue` is described as a "customer SLA tool" here, vs. a "failed cron job store" in ch25. What is the same about both uses, and what is different?

## Summary

| Concept | Key detail |
|---|---|
| `SupportTicket` | Domain object: `ticket_id`, `customer_id`, `org_id`, `subject`, `body`, `priority`, `category` |
| `IntentClassifier` | Keyword heuristic → `billing \| technical \| general \| escalation`; priority order prevents misrouting |
| `TicketRouter` | Maps intent → specialist role; in live mode wraps `MultiAgentOrchestrator` per role |
| `KnowledgeBase` | Seeds `SkillCatalog` with support articles; `build_kb_aware_prompt()` injects relevant articles per intent |
| `CustomerMemory` | Per-customer `MEMORY.md` under `labs/customers/<id>/`; loaded into system prompt for returning customers |
| `SupportAgent` | Combines KB prompt + customer memory + `AIAgent.run_conversation()`; returns `ResolutionResult` |
| `ResolutionResult` | `status` (resolved/escalated/pending) + `response` + `csat_score` (1.0–5.0 or None) |
| `instrument_turn` | Context manager wiring ch23 logs + metrics around every ticket resolution |
| `harness_support_csat_total` | New Prometheus counter — CSAT score sum per category; enables avg CSAT dashboarding |
| `SUPPORT_BENCHMARK_TASKS` | 10 canonical tickets with `success_fn`; covers billing/technical/general/escalation |
| `enrich_kb_from_results()` | CSAT-gated KB enrichment — simulates `LearningLoop.maybe_write_skill()` with quality gating |
| `DeadLetterQueue` as SLA tool | Failed tickets are preserved for replay; SLA breach = ticket age exceeds priority SLA threshold |
| Self-improvement flywheel | Generate tickets → resolve → benchmark → collect → enrich KB → re-benchmark → CSAT ↑ |

---

### Capstone C certification checklist

- [ ] `TicketGenerator.emit_batch()` produces tickets with `ticket_id`, `customer_id`, `org_id`
- [ ] `IntentClassifier.classify()` returns one of `billing / technical / general / escalation`
- [ ] `KnowledgeBase.seed()` writes at least 6 SKILL.md articles to `labs/skills/`
- [ ] `CustomerMemory.update()` writes `MEMORY.md` to `labs/customers/<id>/`
- [ ] `SupportAgent.resolve()` returns a `ResolutionResult` with `status` and `csat_score`
- [ ] `instrument_turn` emits structured JSON logs for every ticket resolution
- [ ] `SUPPORT_BENCHMARK_TASKS` contains exactly 10 tasks across 4 categories
- [ ] `run_support_benchmark()` returns `list[TaskResult]` with 10 entries
- [ ] `collect_support_trajectories()` writes `labs/support_trajectories.jsonl`
- [ ] `enrich_kb_from_results()` writes at least one enriched SKILL.md
- [ ] `ProviderFallbackChain` falls through to secondary when primary circuit trips
- [ ] Failed tickets land in `DeadLetterQueue`
- [ ] `BenchmarkMetrics.render_trend()` shows at least 2 runs

---

### Complete Harness Agent tutorial certification (ch00–ch28)

- [ ] **ch00–ch02**: Environment setup; first API call; provider abstraction
- [ ] **ch03–ch05**: Agent loop; tool registry; observations and recovery
- [ ] **ch06–ch09**: Session storage; prompt assembly; skills; memory
- [ ] **ch10–ch13**: Learning loop; context compression; subagents; MCP integration
- [ ] **ch14–ch17**: Cron scheduler; gateway + CLI; provider resolution; terminal backends
- [ ] **ch18–ch21**: ACP integration; trajectories; plugins; full system integration
- [ ] **ch22**: Multi-agent orchestration (GraphAgent, MultiAgentOrchestrator)
- [ ] **ch23**: Production observability (logs, traces, metrics, health probes)
- [ ] **ch24**: Scalable multi-tenant architecture (TenantContext, RateLimiter, SessionRouter)
- [ ] **ch25**: Reliability and resilience (CircuitBreaker, retry, fallback chain, DLQ)
- [ ] **ch26**: Capstone A — SRE incident response bot (all 25 chapters wired)
- [ ] **ch27**: Capstone B — self-improving agent pipeline (benchmark flywheel)
- [ ] **ch28**: Capstone C — customer support agent with self-improvement (synthesis)

**Congratulations — you have completed the Harness Agent tutorial series.**

You have built agents that:
- Handle real-world domains (SRE operations, customer support)
- Instrument themselves (observability triad)
- Survive provider failures (circuit breakers, fallback chains)
- Scale to multiple tenants (per-org isolation, rate limiting)
- Learn from their own successes (trajectory export → skill extraction → KB enrichment)
- Improve autonomously over time (nightly benchmark flywheel)

The architecture is yours to extend. Every new domain is a new capstone.
