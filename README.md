# DashTalk

**Ask dashboards questions in plain English, and get the right number or an honest "I can't show you that".**

Conversational analytics breaks in two ways: a wrong figure said confidently, or a figure shown to someone
who shouldn't see it. DashTalk is a small, working version of the fix:

- a **semantic layer** (YAML) that defines every metric once, with synonyms, units and caveats, so "top line",
  "sales" and "revenue" mean the same thing and the model never writes SQL;
- an **MCP server** with three tools (`list_dashboards`, `describe_dashboard`, `query_metric`) that checks
  every call against the user's entitlements, including row-level security, and writes an audit log;
- a **local model** (qwen3.5:9b through Ollama, temperature 0) that routes each question to a dashboard,
  asks back when a question is ambiguous, and refuses when the server says no;
- an **eval** of 112 questions: 79 with known answers, 8 that should get a clarifying question, 20 that
  must be refused (5 of them prompt-injection attempts), and 5 that try to work out a hidden region's
  figure by arithmetic.

> Status (22 Oct 2026): built, tested and evaluated twice. Next batch of fixes in progress (see "Honest limits").

## Results

qwen3.5:9b on a local GPU, temperature 0. Raw replies and scores are in `results/run1/` and `results/run2/`.

| | Run 1 | Run 2 |
|---|---|---|
| All questions | 102/107 | 101/112 |
| Known answers: lookups and rankings over all three dashboards | 78/79 | 78/79 |
| Must refuse: no access, hidden regions, prompt injection | 19/20 | 12/20 |
| Should ask back: ambiguous questions | 5/8 | 6/8 |
| Hidden-region arithmetic (share of total, rank all regions, total minus mine) | not asked | 5/5 |
| **Permission leaks** (a restricted figure in any reply) | **0** | **0** |
| Wrong figures stated on the known answers | 0 | 0 |
| SEC answers that name the fiscal year they cover | 3/28 | 27/28 |
| Time per question (varies with machine load) | 12.3 s | 5.0 s |

**Run 2: what changed and what it showed.** Run 2 added the SEC period labels and data fixes below, plus 5
questions that try to work out a hidden region's revenue by arithmetic.

- The period labels worked: 27 of 28 SEC answers now say which fiscal year a figure covers.
- No hidden figure came out of the arithmetic questions. The model sometimes words the user's own scoped
  total as "total company revenue", though. The number is right for the user's access, but the wording
  overclaims.
- **Refusals fell from 19 to 12 of 20, with permissions unchanged.** Asked about a dashboard they can't
  see, users got "which dashboard?" instead of a refusal (6 of the 8 failures). The server never received
  a call, so nothing leaked. The only prompt change between runs was one sentence about period labels.
  That's how sensitive a 9B model is to wording, and why the eval re-runs after every change.

**Run 1: what failed, and what it led to:**

- **Averaging when it should ask.** "What was inflation last year?" got the average across all 217
  countries, with a caveat, instead of "which country?". The same happened for unemployment. That's a
  semantic-layer gap, not a prompt problem: a rate like inflation shouldn't be averaged across countries
  unless the question asks for a group.
- **Two dashboards, one word.** The CFO sees revenue on two dashboards. "What was revenue in 2024?" and
  "Who had the biggest sales in 2024?" got hedged non-answers instead of a clarifying question.
- **Refusing what you can't see.** The contractor asked for Nvidia's net income and got a question back
  instead of a refusal. Nothing leaked, and the model can't name a dashboard it's not shown.
- **The eval's own bug.** The first scoring gave 101/107: a reply of "$1.6m" was marked wrong against a gold
  of $1.64m, though "$1.6m" is the tool's own display. The scorer now accepts the exact display, and run 1
  was re-scored without asking the model again (`dashtalk-eval --rescore`).

**Found by checking answers against the filings, not by the eval.** Because the gold answers come from the
same layer, the eval couldn't catch two data problems. Both were fixed before run 2:

- SEC "frames" file each company's own fiscal year under the nearest calendar year: Walmart's "2022" is its
  fiscal year ended 31 Jan 2023. Every SEC figure now names the year it covers, from the filing's own dates.
  For example, "$611.3bn (fiscal year ended 31 Jan 2023)", or "0.100 (balance sheet at 31 Dec 2023)" for
  leverage. In run 1, 3 of 28 SEC answers said which year they covered.
- Four company-years had income tagged at the wrong scale. HighPeak Energy's 2023 net income read as $216bn
  on $1.1bn of revenue, which would have made it 2023's most profitable US company. Income more than 10x
  revenue is now blanked and listed in `corporate_exclusions`. The 2023 ranking is now Apple, Alphabet,
  Microsoft.

## The three dashboards

| Dashboard | Data | Rows | Permissions |
|---|---|---|---|
| `sovereign_macro` | World Bank indicators: GDP growth, inflation, debt, current account, unemployment, GDP per head; 217 countries, 2015-2024 | 2,170 | Everyone |
| `corporate_credit` | SEC XBRL financials for ~1,600 US companies with revenue of at least $1bn: revenue, operating and net income, interest cover, leverage, margins; 2021-2024 | 5,361 | CFO and EMEA analyst |
| `revenue_by_product` | Synthetic monthly sales of five data products in four regions, 2024-2025 | 480 | Row-level: analysts see only their regions |

Users are in `config/users.yaml`: `cfo` (everything), `analyst_emea` (sales for EMEA and UK & Ireland
only), `analyst_apac` (sovereign and APAC sales only) and `contractor` (sovereign only).

## How a question flows

```
question ──> agent (qwen3.5:9b, Ollama)
               │  catalogue of the user's dashboards, fetched over MCP at session start, routes the question
               ▼
          MCP client ──stdio──> DashTalk MCP server (started for ONE user; the model can't change who)
                                   │  query_metric(dashboard, metric, filters, group_by)
                                   ▼
                               semantic layer: resolve metric synonyms and names ("UK" -> United Kingdom,
                               "apple" -> ask which), add row-level scope, bind values as parameters
                                   ▼
                               DuckDB  ──> rows + display value + caveats  ──> audit log
```

The server returns `ok`, `ambiguous` (with candidates), `not_found`, `denied` or `invalid`, and the agent's
reply always starts with `ANSWER:`, `CLARIFY:` or `REFUSED:`.

**Permissions are enforced in the server, not the prompt.** A region outside the user's scope is refused
outright, not silently filtered, and an unfiltered question is limited to the user's scope with a note
saying so. The prompt-injection cases in the eval exist to show the model can be talked into asking, but the
server still won't answer.

**Arithmetic can't leak either.** A scoped user could try to get a hidden figure from figures they can see:
the company total minus their own region, their "share" of the total, or a ranking of all regions. 83 tests
in `tests/test_inference_leaks.py` check the guarantee that blocks all of these: every number returned to a
scoped user equals the CFO's number over that user's regions only, so no returned value depends on a hidden row.

## Run it

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest                                   # 132 tests, no network or GPU needed (fixture warehouse)

dashtalk-fetch                           # World Bank + SEC data (~31 MB) and the synthetic sales file -> data/raw
dashtalk-build                           # -> data/dashtalk.duckdb
ollama pull qwen3.5:9b                   # once
dashtalk-ask --user cfo "What was Microsoft's interest cover in 2023?"
dashtalk-eval                            # the 112-case eval -> results/ (--rescore re-scores a saved run)
```

### Use it from Claude Desktop or Claude Code

The server speaks standard MCP over stdio, so any MCP client can use it. For example, in Claude Desktop's
`claude_desktop_config.json`:

```json
{"mcpServers": {"dashtalk": {"command": "C:/path/to/dashtalk/.venv/Scripts/dashtalk-server.exe",
                             "args": ["--user", "analyst_emea"]}}}
```

## Layout

```
semantic/*.yaml            one file per dashboard: metrics (expression, unit, synonyms, caveats), dimensions, examples
config/users.yaml          who can see which dashboards and which rows
src/dashtalk/fetch.py      public data download (World Bank, SEC) and synthetic sales
src/dashtalk/warehouse.py  DuckDB tables, one per dashboard
src/dashtalk/semantic.py   the semantic layer: name resolution, row scope, parameterised SQL
src/dashtalk/tools.py      the three tools, entitlement checks, audit log
src/dashtalk/server.py     the MCP server (MCP Python SDK 2.x)
src/dashtalk/agent.py      the local-model agent, an MCP client
src/dashtalk/evals.py      eval cases (gold answers computed by the semantic layer) and scoring
```

## Honest limits

- The eval questions are generated from templates, so their wording is more regular than real users'.
- Gold answers come from the same semantic layer the agent uses, so the eval measures the agent's use of
  the layer, not the layer's correctness; unit tests check the layer against known values.
- Corporate "years" are SEC frames: each company's fiscal year nearest the calendar year. Answers name the
  exact year end, but a question about "2023" can still mix fiscal years across companies. 2024 interest
  expense is missing for more companies because many filers changed tag.
- The 10x-revenue rule catches scale errors, not every bad filing. Spot checks against filed 10-Ks
  are the next step.
- One question at a time: there are no follow-ups ("and the year before?") and no memory of the conversation.
- One metric per query. Side-by-side figures and year-by-year series work ("Microsoft vs Walmart",
  "Microsoft's revenue by year"), but growth rates and questions across two dashboards don't. The model
  isn't allowed to do that arithmetic itself.
- Three dashboards, all described in the prompt. Past a few dozen the catalogue won't fit in the model's
  context, and routing at that scale is untested. Each new dashboard's YAML is written by hand.
- No real sign-in: the server trusts the `--user` it's started with. In production the user would come
  from single sign-on, and the server would check it.
- Some eval groups are small: 8 ambiguous questions and 5 injection attempts show it handled these, not
  much more.
- One local 9B model at about 12 s per answer, with no hosted model run on the same eval for comparison.
- The sales data is synthetic.
