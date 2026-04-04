# User Experience

qracer user experience design — natural language interface and Agent collaboration.

---

## 1. Interface Philosophy

**Natural language first, commands as exception**

Users speak naturally, Agent understands and acts.

| Interface | Use Case | Example |
|-----------|----------|---------|
| **Natural Language** | All investment analysis, questions, exploration | "Analyze AAPL", "How's the semiconductor sector?" |
| **Commands** | System management only | `/status`, `/config` |

---

## 2. Natural Language Interface

### Input Patterns

**Ticker Mention**
```
"How's AAPL?"
"Show me Apple stock"
"Compare TSLA and NVDA"
```

**Sector/Theme Mention**
```
"Analyze semiconductor sector"
"What AI stocks are there?"
"Tell me about EV battery theme"
```

**Portfolio Related**
```
"Check my portfolio risk"
"How's my portfolio doing?"
"Risk check my new position"
```

**Complex Questions**
```
"Any semiconductor stocks with PE under 20 worth buying?"
"What correlates highly with my portfolio?"
"Which stocks report earnings next week?"
```

### Agent Response Patterns

**When Clarification Needed**
```
User: "Recommend stocks to buy"
Agent: "Which sector or theme? Or search the whole market?"
```

**When Additional Info Needed**
```
User: "How's this stock?"
Agent: "Which stock? Recently mentioned: AAPL, TSLA, NVDA?"
```

**When Ambiguous**
```
User: "Looks good"
Agent: "AAPL, right? Analysis from buy perspective?"
```

---

## 3. Conversation Context

### 3.1 Technical Implementation

**Memory Structure**
```
Session Memory (Redis/SQLite)
├── session_id: uuid
├── user_id: telegram_id
├── created_at: timestamp
├── last_activity: timestamp
├── context_ttl: 600s (10min)
│
├── current_topic:
│   ├── type: ticker | sector | theme | portfolio
│   ├── value: "AAPL" | "semiconductors" | "ai"
│   └── mentioned_at: timestamp
│
├── topic_stack: [  # last 5
│   {type, value, mentioned_at},
│   ...
│]
│
├── analysis_state:
│   ├── depth: quick | deep
│   ├── intent: buy | sell | research | monitor
│   └── pending_questions: [string]
│
└── user_preferences:
    ├── default_analysis_depth: quick
    ├── alert_thresholds: {...}
    └── preferred_sectors: [...]
```

**Storage Policy**
| Data | Storage | TTL | Recovery |
|------|---------|-----|----------|
| Active Session | Redis | 10min | Auto |
| Topic History | SQLite | 30days | On session resume |
| User Preferences | SQLite | Permanent | On login |

**Expiration & Renewal**
- Update `last_activity` on every message
- 10min no response → pending state (recoverable)
- 24hrs no response → archive session (history kept)

### 3.2 User Experience Flow

**Normal Flow Example**
```
User:  "Analyze AAPL"
Agent: [Provides AAPL deep analysis]
       Current: AAPL | Depth: deep

User:  "PE looks high?"
Agent: [Explains AAPL PE context]
       Current: AAPL | Depth: deep (maintained)

User:  "When's good entry timing?"
Agent: [Analyzes AAPL entry points]
       Current: AAPL | Depth: deep (maintained)

User:  "How about TSLA?"
Agent: [Provides TSLA deep analysis]
       Current: TSLA | Stack: [TSLA, AAPL]
```

**Context Switch Example**
```
User:  "Analyze AAPL"
Agent: [Analysis] Current: AAPL

User:  "Show me real-time summary"  # intent switch
Agent: "Switch AAPL to real-time summary? Or different stock?"

User:  "No, AAPL is right"
Agent: [AAPL real-time summary] Depth: quick
```

### 3.3 Complex Scenario Handling

**Multiple Topics Mixed**
```
User: "Which is better, AAPL or TSLA?"
→ Stack: [AAPL, TSLA] (equal priority)
→ Comparison mode activated

User: "But NVDA also looks good"
→ Stack: [NVDA, AAPL, TSLA]
→ "Compare all three? Or just add NVDA analysis?"
```

**Ambiguous Pronouns**
```
User: "Analyze AAPL"
Agent: [Provides analysis]

User: "Is this a buy?"
→ "this" = AAPL (clear)

User: "Check TSLA too"
Agent: [TSLA analysis]

User: "This looks risky"
→ "This" = TSLA (last mentioned)
→ But: consider context → "TSLA's risk you're referring to?"
```

**Conflict Resolution**
```
User: "Any good semiconductor stocks?"
Agent: [Semiconductor sector analysis] Current: semiconductors

User: "What about AAPL?"  # sector → ticker switch
→ Stack: [AAPL, semiconductors]
→ "View AAPL in semiconductor context? Or as individual stock?"
```

### 3.4 Agent Decision Logic

**Context Retention Decision Tree**
```
New message arrives
├── Ticker/sector mentioned?
│   ├── Yes → Switch to that topic
│   │         └── Save previous to stack
│   └── No → Maintain current topic
│
├── Pronoun present?
│   ├── "this/it/this stock" → last ticker
│   ├── "previous one" → search topic stack
│   ├── "another one" → suggest from stack excluding current
│   └── Ambiguous → clarification question
│
├── Intent change detected?
│   ├── "quickly" → switch to quick
│   ├── "in detail" → switch to deep
│   └── "compare" → comparison mode
│
└── 10min elapsed?
    ├── Yes → "We were discussing AAPL. Continue?"
    └── No → Silent recovery
```

**Reset Conditions**
| Condition | Action |
|-----------|--------|
| `/reset` command | Clear context immediately |
| 24hrs elapsed | Archive session, start new |
| "Let's start fresh" | Clear context + confirm |
| Topic completely changed (sector↔portfolio) | Smooth transition + save to stack |

### 3.5 Unknown Topic Handling

**Search Priority**
```
Unknown topic "XYZ" appears
  ↓
1. Local Context (instant, <100ms)
   - Current topic stack
   - Session memory
   
2. Embedding Search (fast, <500ms)
   - Past conversation history
   - Stored topic metadata
   
3. Web Search (slow, 1-3s)
   - Ticker/sector definitions
   - Latest news/info
   
4. Admit
   - "I'm not sure"
```

**Handling Example**
```
User: "What's QuantumScape?"
  ↓
[Local/embedding search] None
[Web search] "QuantumScape is solid-state battery development company..."
  ↓
Agent: "QuantumScape develops solid-state batteries. 
        Recent stock price... [web search summary + ticker info]"
```

**Response Patterns**
| Situation | Response |
|-----------|----------|
| Found in local/embedding | Direct answer + source mention ("As you mentioned before...") |
| Found via web search | Summary answer + "According to search..." |
| Not found | "I'm not sure. Can you tell me more?" |

---

## 4. System Commands

System management commands only. Investment analysis via natural language.

| Command | Description |
|---------|-------------|
| `/status` | Check system status |
| `/config` | View/change settings |
| `/help` | Help |

---

## 5. Dashboard

qracer main interface. Left sidebar menu + right info panel structure.

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  qracer                                                 │
├──────────┬──────────────────────────────────────────────┤
│ (sidebar)│              (main panel)                    │
│          │                                              │
│  DASH    │  - Display info based on selected menu       │
│  Overview│                                              │
│  Portfolio│                                             │
│  Watchlist│                                             │
│  Alerts   │                                             │
│          │                                              │
│  CHAT    │                                              │
│  New Chat│                                              │
│  History │                                              │
│          │                                              │
│  SETTINGS│                                              │
│  General │                                              │
│  Providers│                                             │
│  Notifications│                                          │
│          │                                              │
└──────────┴──────────────────────────────────────────────┘
```

### Sidebar Menu

**DASH Section**
| Menu | Content | Real-time Update |
|------|---------|------------------|
| Overview | Portfolio summary + market overview + alerts status | 5s |
| Portfolio | Holdings detail (P&L, weight, risk) | 5s |
| Watchlist | Watchlist (price, change%) | 5s |
| Alerts | Active alerts list + trigger history | Instant |

**CHAT Section**
| Menu | Function |
|------|----------|
| New Chat | Start new session with Agent (luce) |
| History | Past conversation list, search, resume |

**SETTINGS Section**
| Menu | Settings |
|------|----------|
| General | Theme, language, default analysis depth |
| Providers | Data sources (Finnhub, FRED, etc.) API key management |
| Notifications | Notification channels (Telegram, Email), threshold settings |

### Main Panel Components

**Overview Example**
```
┌────────────────────────────────────────┐
│  Portfolio Summary                     │
│  AAPL  $175.20  +1.2%   $12,450        │
│  TSLA  $242.50  -0.8%   $24,250        │
│  NVDA  $875.10  +2.5%   $43,755        │
│  ─────────────────────────────────     │
│  Total: $80,455  |  Day: +$1,245       │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Market Overview                       │
│  S&P 500   5,234.12  +0.4%            │
│  NASDAQ   16,432.50  +0.7%            │
│  VIX         14.20   -2.1%            │
└────────────────────────────────────────┘
```

### Interaction Patterns

**Dashboard → Chat Transition**
```
1. Click AAPL in Overview
2. Auto opens New Chat
3. Agent: "Tell you about AAPL?"
4. User types naturally to continue conversation
```

**Chat Referencing Dashboard**
```
User: "Check my portfolio risk"
Agent: [Provides portfolio risk analysis]
      "You can also check real-time risk gauge in Overview tab."
```

---

## 6. Visualization

### Output Format Selection

| Situation | Format | Characteristics |
|-----------|--------|-----------------|
| Quick check | Summary Card | Under 3 lines, essentials only |
| Detailed analysis | Structured Report | Sectioned |
| Comparison | Side-by-side Table | Multiple tickers |
| Timeline | Timeline | Event sequence |

### Mobile Optimization

- 80 character width standard
- Tickers in **bold**
- Thousands separators for numbers
- Color symbols: 🔴 🟡 🟢

---

## 7. Install (qracer install)

### Requirements

- Python 3.10+
- pip or uv

### Install Command

```bash
# Install with pip
pip install qracer

# Install with uv (recommended, faster)
uv add qracer
```

### First Run (qracer install)

After install, run `qracer install` for initial setup:

```
$ qracer install

🎯 Starting qracer installation.

1. Create config directory
   Creating ~/.qracer/... [OK]

2. Data source setup
   Enter Finnhub API key (optional): 
   Enter FRED API key (optional):

3. Telegram notification setup (optional)
   Bot Token: 
   Chat ID:

4. Portfolio initial setup
   Base currency (USD/KRW): USD

✅ Install complete! Check commands with 'qracer --help'.
```

### Post-Install First Run

```bash
$ qracer

🎯 Welcome to qracer!

I'm luce, your trading research partner.

Get started:
• "Analyze AAPL" — Deep stock analysis
• "Check my portfolio" — Portfolio management
• "How's semiconductor sector?" — Sector analysis

How can I help?
```

### Progressive Feature Introduction

| Stage | Condition | Introduction |
|-------|-----------|--------------|
| 1 | First use | 3 basic question patterns |
| 2 | 5+ conversations | Alert settings, portfolio features |
| 3 | Portfolio created | Risk module, position sizing |
| 4 | 20+ analyses | Advanced filters, complex questions |

---

## 8. Agent Collaboration Principles

### Can Do

- Real-time and historical data lookup
- Technical/fundamental analysis
- Risk assessment and position sizing
- Portfolio monitoring and alerts
- Sector/theme analysis

### Cannot Do (Explicit)

- Execute actual trades
- Investment advice or guarantees
- Future price predictions

### When Uncertain

> "I'm not sure. Need more research."

Admit limitations rather than guess.

---

## 9. Design Principles

1. **Natural Language First** — Speak naturally without memorizing
2. **Context Awareness** — Remember and continue conversation flow
3. **Progressive Disclosure** — Introduce complex features gradually
4. **Transparency** — Clear about limitations, no guessing
5. **Fail Gracefully** — Suggest next action on errors
