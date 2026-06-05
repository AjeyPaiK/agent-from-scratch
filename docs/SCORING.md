# Code-Based Scoring (Assignment §2.1)

This document specifies a deterministic, oracle-based evaluation protocol in
which no large language model serves as judge. All scores are **turn-scoped**:
the evaluation of turn $t$ depends exclusively on the tool invocations and
guardrail verdicts recorded during that turn, and carries no dependence on
outcomes from preceding turns $t-1, t-2, \ldots$.

**Implementation:** `scoring/compare.py`, `scoring/trustworthiness.py`,
`scoring/oracle/`, `observability/langfuse_integration.py`.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $a$ | Actual tool output (JSON dict from runtime tool) |
| $e$ | Expected output from CSV oracle (`scoring/oracle/`) |
| $\mathbb{1}[P]$ | Indicator: $1$ if predicate $P$ is true, else $0$ |
| $w_f$ | Non-negative weight for field/component $f$ (sums to $1$ per tool) |
| $A \triangleq B$ | Boolean equality of truthiness: $\mathbb{1}[\mathrm{bool}(A)] = \mathbb{1}[\mathrm{bool}(B)]$ |

### Jaccard similarity (set overlap)

For finite sets $S_a, S_e$:

$$
J(S_a, S_e) =
\begin{cases}
1 & \text{if } S_a = \varnothing \text{ and } S_e = \varnothing \\[4pt]
0 & \text{if exactly one of } S_a, S_e \text{ is empty} \\[4pt]
\dfrac{|S_a \cap S_e|}{|S_a \cup S_e|} & \text{otherwise}
\end{cases}
$$

**Why Jaccard?** Annex entries, limit rows, and labelling obligation strings are
*sets of claims*. Partial overlap (right entries, missing one row) should score
between $0$ and $1$, not collapse to all-or-nothing.

### String normalization (labelling lists)

For list field $\ell$:

$$
\mathrm{norm}(\ell) = \{\, \mathrm{lowercase}(\mathrm{strip}(x)) \mid x \in \ell,\; x \neq \text{""} \,\}
$$

---

## 2. Per-tool accuracy

Each tool call $i$ in turn $t$ receives an accuracy $s_i \in [0,1]$ by
comparing $a_i$ to oracle $e_i$. If scoring is skipped (§2.4), $s_i$ is
undefined and excluded from averages.

**Hard zero (all tools):**

$$
s_i = 0 \quad \text{if } a_i.\texttt{error} \text{ is present}
$$

---

### 2.1 `lookup_ingredient_regulation`

**Weights** (sum to $1$):

$$
w_{\mathrm{found}} = 0.20,\;
w_{\mathrm{status}} = 0.25,\;
w_{\mathrm{entries}} = 0.30,\;
w_{\mathrm{fields}} = 0.20,\;
w_{\mathrm{match}} = 0.05
$$

**Entry key** for annex row $r$:

$$
\kappa(r) =
\begin{cases}
(\text{annex id},\; \text{entry number}) & \text{if parsed from ``Annex X, entry Y''} \\
(\texttt{annex\_reference},\; \texttt{legal\_status}) & \text{fallback}
\end{cases}
$$

**Entry-field match** for key $k$: $\phi_k = 1$ iff actual and expected
entries with key $k$ agree on:

- `legal_status` (exact)
- `substance_name` (stripped string equality)
- `conditions` (list length + four condition subfields, stripped equality)

Let $K_a, K_e$ be the sets of keys from actual/expected `annex_entries`, and
$n = \max(|K_e|, 1)$:

$$
m_{\mathrm{fields}} = \frac{1}{n} \sum_{k \in K_e} \phi_k
$$

**Raw score** (before match-type scaling):

$$
S_{\mathrm{raw}} =
\begin{cases}
0 & a.\texttt{error} \\[6pt]
w_{\mathrm{found}} \cdot \mathbb{1}[a.\texttt{found} \triangleq e.\texttt{found}]
& \text{if } \neg e.\texttt{found} \quad\text{(early exit)} \\[8pt]
w_{\mathrm{found}} \cdot \mathbb{1}[a.\texttt{found} \triangleq e.\texttt{found}] \\
\quad + w_{\mathrm{status}} \cdot \mathbb{1}[a.\texttt{overall\_status} = e.\texttt{overall\_status}] \\
\quad + w_{\mathrm{entries}} \cdot J(K_a, K_e) \\
\quad + w_{\mathrm{fields}} \cdot m_{\mathrm{fields}} \\
\quad + w_{\mathrm{match}} \cdot \mathbb{1}[a.\texttt{match\_type} = e.\texttt{match\_type}]
& \text{if } e.\texttt{found}
\end{cases}
$$

**Match-type confidence** (penalizes fuzzy identification):

$$
\gamma(\tau) =
\begin{cases}
1.00 & \tau \in \{\texttt{exact}, \texttt{cas}\} \\
0.85 & \tau = \texttt{identified} \\
0.70 & \tau = \texttt{fuzzy} \\
1.00 & \text{otherwise (unknown type)}
\end{cases}
$$

**Final lookup accuracy:**

$$
s_{\mathrm{lookup}} = \min\!\left(1,\; S_{\mathrm{raw}} \cdot \gamma(a.\texttt{match\_type})\right)
$$

#### When is the score exactly $1$?

All of: no `error`; `found` agrees; if $e.\texttt{found}$, then
`overall_status` matches, $J(K_a,K_e)=1$, every $\phi_k=1$, `match_type`
matches, and $\gamma=1$.

#### When is the score exactly $0$?

- `error` present, **or**
- `found` disagrees with oracle **and** $e.\texttt{found}=\texttt{true}$ (no
  positive weights are earned; $S_{\mathrm{raw}}=0$).

#### Partial scores (examples)

| Situation | Formula fragment | Typical value |
|-----------|------------------|---------------|
| Both agree ingredient not found | $w_{\mathrm{found}} \cdot 1$ | $0.20$ |
| Correct `found`, wrong `overall_status` | lose $w_{\mathrm{status}}$ | $\leq 0.75$ before $\gamma$ |
| Correct keys, one entry field wrong | $m_{\mathrm{fields}} = \frac{n-1}{n}$ | lose up to $w_{\mathrm{fields}}/n$ |
| Missing annex entry | $J(K_a,K_e) < 1$ | lose up to $w_{\mathrm{entries}}$ |
| Fuzzy match otherwise perfect | multiply by $0.70$ | $\leq 0.70$ |

**Why partial?** Regulatory lookup failures are often *localized* (wrong summary
status, correct annex rows). Weighted decomposition localizes the failure for
debugging and RL reward shaping.

---

### 2.2 `check_concentration_compliance`

**Weights:**

$$
w_{\mathrm{found}} = 0.15,\;
w_{\mathrm{cat}} = 0.10,\;
w_{\mathrm{limits}} = 0.45,\;
w_{\mathrm{compliant}} = 0.30
$$

**Limit row key:**

$$
\lambda(r) = (\texttt{annex\_reference},\; \texttt{product\_type\_body\_parts})
$$

**Limit-field match** $\psi_k$: for row key $k$, compare
`max_concentration_text`, `max_concentration_percent`, `other_conditions`,
`compliant` (exact equality).

Let $L_a, L_e$ be limit-row key sets, $n_\ell = \max(|K_e^{\mathrm{lim}}|, 1)$
where $K_e^{\mathrm{lim}} = \{\lambda(r) : r \in e.\texttt{limits}\}$:

$$
m_{\mathrm{lim}} = \frac{1}{n_\ell} \sum_{k \in K_e^{\mathrm{lim}}} \psi_k
$$

**Limits sub-score** $L$:

$$
L =
\begin{cases}
1 & |e.\texttt{limits}| = |a.\texttt{limits}| = 0 \\[4pt]
\frac{1}{2} & \text{both empty but } a.\texttt{message} \neq e.\texttt{message} \\[6pt]
\dfrac{J(L_a, L_e) + m_{\mathrm{lim}}}{2} & \text{otherwise}
\end{cases}
$$

**Compliant sub-score** $C$:

$$
C =
\begin{cases}
\mathbb{1}[a.\texttt{compliant} = e.\texttt{compliant}]
& \text{if } e.\texttt{compliant} \neq \texttt{null} \\[4pt]
1 & \text{if } e.\texttt{compliant} = \texttt{null} \text{ and } a.\texttt{compliant} = \texttt{null} \\[4pt]
0 & \text{otherwise}
\end{cases}
$$

**Product category** earns $w_{\mathrm{cat}}$ only when
$e.\texttt{product\_category}$ is non-empty and strings match (stripped).

**Final concentration accuracy:**

$$
s_{\mathrm{conc}} =
\begin{cases}
0 & a.\texttt{error} \\[4pt]
w_{\mathrm{found}} \cdot \mathbb{1}[a.\texttt{found} \triangleq e.\texttt{found}]
& \text{if } \neg e.\texttt{found} \\[6pt]
\min\!\Big(1,\;
w_{\mathrm{found}} \cdot \mathbb{1}[a.\texttt{found} \triangleq e.\texttt{found}] \\
\quad + w_{\mathrm{cat}} \cdot \mathbb{1}[\texttt{cat}(a) = \texttt{cat}(e)] \\
\quad + w_{\mathrm{limits}} \cdot L \\
\quad + w_{\mathrm{compliant}} \cdot C
\Big)
& \text{if } e.\texttt{found}
\end{cases}
$$

#### Partial scores (examples)

| Situation | Effect |
|-----------|--------|
| Both not found | $s = 0.15$ |
| Found, wrong limit key set | $L < 1$; lose up to $0.45 \cdot (1-L)$ |
| Correct keys, wrong `compliant` on one row | $m_{\mathrm{lim}} < 1$; blended into $L$ |
| Correct limits, wrong compliance flag | lose $w_{\mathrm{compliant}} = 0.30$ |
| No concentration supplied ($e.\texttt{compliant}=\texttt{null}$, $a$ also null) | full $w_{\mathrm{compliant}}$ awarded |

**Why partial?** Compliance is a conjunction of *which limits apply* and
*whether the stated % satisfies them* — separate weights reflect distinct failure
modes.

---

### 2.3 `get_labelling_marketing_rules`

**Weights:**

$$
w_{\mathrm{found}} = 0.15,\;
w_{\mathrm{inci}} = 0.10,\;
w_{\mathrm{cat}} = 0.10,\;
w_{\mathrm{lab}} = 0.35,\;
w_{\mathrm{mkt}} = 0.20,\;
w_{\mathrm{ref}} = 0.10
$$

For list fields $f \in \{\texttt{labelling\_requirements},\;
\texttt{marketing\_restrictions},\; \texttt{references}\}$:

$$
S_f = J\!\left(\mathrm{norm}(a.f),\; \mathrm{norm}(e.f)\right)
$$

**Final labelling accuracy:**

$$
s_{\mathrm{lab}} =
\begin{cases}
0 & a.\texttt{error} \\[4pt]
w_{\mathrm{found}} \cdot \mathbb{1}[a.\texttt{found} \triangleq e.\texttt{found}]
& \text{if } \neg e.\texttt{found} \\[6pt]
\min\!\Big(1,\;
w_{\mathrm{found}} \cdot \mathbb{1}[a.\texttt{found} \triangleq e.\texttt{found}] \\
\quad + w_{\mathrm{inci}} \cdot \mathbb{1}[\texttt{strip}(a.\texttt{inci\_name}) = \texttt{strip}(e.\texttt{inci\_name})] \\
\quad + w_{\mathrm{cat}} \cdot \mathbb{1}[\texttt{cat}(a) = \texttt{cat}(e)] \\
\quad + w_{\mathrm{lab}} \cdot S_{\mathrm{labelling\_requirements}} \\
\quad + w_{\mathrm{mkt}} \cdot S_{\mathrm{marketing\_restrictions}} \\
\quad + w_{\mathrm{ref}} \cdot S_{\mathrm{references}}
\Big)
& \text{if } e.\texttt{found}
\end{cases}
$$

#### Partial scores (examples)

| Situation | Effect |
|-----------|--------|
| Both not found | $s = 0.15$ |
| 4 of 5 obligation strings match | $S_{\mathrm{lab}} = 0.8$; lose $0.35 \cdot 0.2 = 0.07$ |
| Extra marketing line not in oracle | $J < 1$ on marketing set |
| Wrong `product_category` | lose $w_{\mathrm{cat}} = 0.10$ |

**Why partial?** Labelling output is an *ensemble of obligation strings*; Jaccard
rewards near-complete lists without requiring bitwise identical ordering.

---

### 2.4 Skipped tool calls (no $s_i$)

`score_tool_call` returns $\varnothing$ when:

$$
a_i.\texttt{guardrail\_blocked} = \texttt{true}
\;\vee\;
a_i \notin \texttt{dict}
$$

Blocked calls remain in the trace but are **excluded** from
$\mathcal{S}_t$ (the set of scored accuracies in turn $t$).

---

## 3. Turn-level tool accuracy component

Let $\mathcal{E}_t$ = executed (non-blocked) tool calls in turn $t$,
$\mathcal{S}_t = \{\, s_i \mid \text{call } i \text{ scored} \,\}$.

$$
A_t =
\begin{cases}
\dfrac{1}{|\mathcal{S}_t|} \displaystyle\sum_{s \in \mathcal{S}_t} s
& \text{if } \mathcal{S}_t \neq \varnothing \\[10pt]
1 & \text{if } \mathcal{E}_t = \varnothing \quad\text{(no tools run)} \\[6pt]
0 & \text{if } \mathcal{E}_t \neq \varnothing \text{ and } \mathcal{S}_t = \varnothing
\end{cases}
$$

**Interpretation:**

- **$A_t = 1$, no tools:** LLM answered without tools — vacuous success on
  tool correctness.
- **$A_t = 0$, tools ran but unscorable:** e.g. all calls guardrail-blocked;
  tool layer contributed no verifiable evidence.

---

## 4. Guardrail integrity component

Three symbolic stages. For turn $t$, define stage scores in $[0,1]$:

**Pre-input** (input accepted?):

$$
G_{\mathrm{in}} = \mathbb{1}[\text{pre\_input.passed}]
$$

**Pre-tool** (per tool-call argument validation):

$$
G_{\mathrm{tool}} =
\begin{cases}
\dfrac{1}{|\mathcal{V}_{\mathrm{pre\_tool}}|}
  \displaystyle\sum_{v \in \mathcal{V}_{\mathrm{pre\_tool}}} \mathbb{1}[v.\texttt{passed}]
& \text{if } \mathcal{V}_{\mathrm{pre\_tool}} \neq \varnothing \\[8pt]
\text{(omitted from average)} & \text{if no tool calls validated}
\end{cases}
$$

**Post-output** (answer vs tool evidence):

$$
G_{\mathrm{out}} =
\begin{cases}
\mathbb{1}[\text{post\_output.passed}] & \text{if stage ran} \\
\text{(omitted)} & \text{if post\_output is null}
\end{cases}
$$

Let $\mathcal{G}_t$ be the set of stage scores that apply this turn.

**Blocked-at-pre-input** (malicious / out-of-scope / non-EU): correct refusal is
*trustworthy*, so:

$$
G_t =
\begin{cases}
1 & \text{if turn blocked and } G_{\mathrm{in}} = 0 \\[6pt]
\dfrac{1}{|\mathcal{G}_t|} \displaystyle\sum_{G \in \mathcal{G}_t} G
& \text{otherwise}
\end{cases}
$$

where $\mathcal{G}_t = \{G_{\mathrm{in}}\}$ on early-exit blocked turns (only
pre-input ran).

**Langfuse binary scores** (§6) export $\mathbb{1}[\cdot]$ per stage for
dashboards; $G_t$ above is the **composite guardrail integrity** used inside
$T_t$ and can be fractional when only some pre-tool calls fail.

#### Post-output partial contribution

Post-output is binary per turn, but when $G_{\mathrm{out}} = 0$ it drags the
average:

$$
G_t = \frac{G_{\mathrm{in}} + G_{\mathrm{tool}} + 0}{3}
\quad\text{(example: input ok, tools ok, output failed)}
$$

---

## 5. Composite turn trustworthiness

**Weights** (defaults in `scoring/trustworthiness.py`):

$$
\alpha = 0.6 \quad (\text{tool}),\qquad \beta = 0.4 \quad (\text{guardrail}),
\qquad \alpha + \beta = 1
$$

**Composite score** for turn $t$:

$$
\boxed{
T_t = \mathrm{clip}_{[0,1]}\!\left(\alpha \cdot A_t + \beta \cdot G_t\right)
}
$$

where $\mathrm{clip}_{[0,1]}(x) = \min(1, \max(0, x))$.

### Boundary cases

| Case | $A_t$ | $G_t$ | $T_t$ |
|------|---------|---------|---------|
| Perfect tools + all guardrails pass | $1$ | $1$ | $1.0$ |
| Correct pre-input block | $1$ | $1$ | $1.0$ |
| Tools unscorable, guardrails pass | $0$ | $1$ | $0.4$ |
| Perfect tools, post-output fails | $1$ | $\frac{2}{3}$ | $0.6 + 0.4 \cdot \frac{2}{3} \approx 0.87$ |
| Not found only (lookup) | $0.20$ | $1$ | $0.6 \cdot 0.2 + 0.4 = 0.52$ |

**Why this composite?** $A_t$ measures *factual tool correctness* against pinned
regulatory data; $G_t$ measures *symbolic safety and verification*. Weighting
$\alpha > \beta$ prioritizes correct evidence retrieval while still punishing
guardrail failures that indicate hallucination or unsafe I/O.

---

## 6. Langfuse export mapping

Each turn trace receives numeric scores. Relationship to formulas above:

| Langfuse `name` | Value | Formula |
|-----------------|-------|---------|
| `tool_accuracy_lookup` | $s_{\mathrm{lookup}}$ | §2.1 |
| `tool_accuracy_concentration` | $s_{\mathrm{conc}}$ | §2.2 |
| `tool_accuracy_labelling` | $s_{\mathrm{lab}}$ | §2.3 |
| `turn_tool_accuracy_avg` | $A_t$ | §3 (mean over $\mathcal{S}_t$) |
| `guardrail_pre_input` | $G_{\mathrm{in}}$ | §4 |
| `guardrail_pre_tool` | $\min_G G_{\mathrm{tool}}$ or $1$ if no calls | all-or-nothing export |
| `guardrail_post_output` | $G_{\mathrm{out}}$ | §4 (if ran) |
| `guardrails_all_passed` | $\mathbb{1}[\forall \text{ stages pass}]$ | binary aggregate |
| **`turn_trustworthiness`** | **$T_t$** | **§5** |

Trace metadata strings: `tool_accuracy_avg` $\approx A_t$,
`turn_trustworthiness` $\approx T_t$ (4 decimal places).

---

## 7. Oracle independence

Runtime tools call the **live CosIng API** (`data/cosing_api.py`). Oracles read
**pinned CSV snapshots** (`data/annex_snapshots/default/`). Scoring therefore
measures agreement between two independent implementations of the same regulatory
source — not self-consistency of one code path.

---

## 8. Code index

| File | Role |
|------|------|
| `scoring/compare.py` | $s_{\mathrm{lookup}}, s_{\mathrm{conc}}, s_{\mathrm{lab}}$ |
| `scoring/score_tool_call.py` | Oracle dispatch + comparator selection |
| `scoring/trustworthiness.py` | $A_t, G_t, T_t$ |
| `scoring/trace.py` | Attach $s_i$ to `ToolTraceEntry` after exposition |
| `observability/langfuse_integration.py` | Push scores to Langfuse |
