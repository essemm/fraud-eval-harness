# Project Brief: Sequence-Aware Fraud Detection Evaluation Harness

**Status:** Draft · **Owner:** Scott MacGibbon · **Type:** Self-directed portfolio project

This document is a specification. It is written so that the system can be
regenerated from the brief alone — the existing code is one valid
implementation, not the definition. It serves three audiences: stakeholders
deciding whether the approach is sound, engineers (or code-generation tools)
producing the implementation, and test authors deriving test cases from the
acceptance criteria.

---

## 1. Executive summary

Card fraud is not a property of a single transaction. A $4 online purchase is
unremarkable in isolation; six of them in ninety seconds from a device the
cardholder has never used is a card-testing attack. The signal lives in the
*sequence* and in the *deviation from a cardholder's own baseline*, not in any
row read alone.

This project builds an evaluation harness that makes that claim measurable. It
generates synthetic transaction data with known fraud sequences and deliberately
ambiguous legitimate behaviour ("hard negatives"), scores it, and evaluates the
scorer under an explicit **cost model** rather than accuracy. Accuracy is
meaningless at sub-1% fraud rates; a model that flags nothing scores 99.7%. The
harness optimises for the business cost of being wrong, asymmetrically: a missed
fraud and a false alarm do not cost the same.

The deliverable is not a fraud model. It is the apparatus that tells you whether
*any* fraud model is good enough, where it fails, and at what operating point it
should run. The scorer is pluggable: a transparent rule baseline ships first, and
a machine-learning model can be swapped in later behind the same interface and
judged on the same harness.

### Why this design

- **Evaluation-first.** The hard problem in applied fraud detection is not the
  classifier; it is knowing whether it works and at what threshold to run it.
  The harness is the centrepiece, by design.
- **Cost-weighted, not accuracy-weighted.** Decisions are made against a
  configurable cost model so the precision/recall trade-off can be *shown*
  moving, not asserted.
- **Explainable baseline before ML.** Every v1 decision carries a human-readable
  reason. The ML model is a later swap-in measured against this baseline using
  the identical harness — which is itself a clean A/B evaluation story.

---

## 2. Problem statement and goals

### Problem

Given a stream of card transactions, identify accounts experiencing fraud, early
enough to act, without drowning a review team in false alarms. The detector must
distinguish genuine fraud from legitimate behaviour that *looks* like fraud
(travel, large one-off purchases, a new phone).

### Goals

1. Produce a realistic, fully-labelled synthetic dataset in which fraud is
   injected as causal sequences, including hard negatives.
2. Reconstruct per-transaction sequence context and per-cardholder baselines from
   a flat event stream.
3. Score transactions and aggregate to an account-level decision, with every
   decision explainable.
4. Evaluate the scorer under configurable cost models, reporting recall *per
   attack type* and the cost-minimising operating threshold.

### Known limitations

- Not a production fraud system; no real cardholder data, no streaming
  infrastructure, no model serving.
- Not a state-of-the-art classifier. The scorer quality is secondary to the
  harness that measures it.
- No PII, ever. Data is synthetic by construction.

### Success measures

- The harness reports per-scenario recall, precision, and total cost across a
  threshold sweep, and identifies the cost-minimising threshold under each cost
  model.
- The rule baseline catches the majority of each fraud scenario while keeping the
  hard-negative false-positive rate measurably below the naive single-row
  threshold it is compared against.
- Swapping the scorer implementation requires no change to the feature or
  evaluation modules.

---

## 3. Scope and module architecture

The system is four modules connected by CSV interfaces. Each is independently
readable and independently testable. The seams are deliberate: they are where
implementations get swapped and where stakeholders reason about the system.

```
generate_synthetic.py  ──>  transactions.csv  ──┬──────────────┐
                            (native currency)    │              │
                                                 v              v
                       fx_rates.csv  ──────>  profile.py    features.py  ──>  score.py  ──>  evaluate.py
                                                  │              ^
                                                  └──> card_profiles.csv ──┘
```

Both `profile.py` and `features.py` read `fx_rates.csv` and normalise native
amounts to USD via the shared `fx.to_usd` helper.

| Module | Responsibility | Reads | Writes |
|---|---|---|---|
| `generate_synthetic.py` | Emit labelled synthetic transactions (native currency) with injected fraud sequences and hard negatives | — | `transactions.csv` |
| `fx.py` | Shared currency→USD conversion helper | — | — (library) |
| `profile.py` | Normalise to USD; aggregate transactions into a per-card baseline dimension | `transactions.csv`, `fx_rates.csv` | `card_profiles.csv` |
| `features.py` | Normalise to USD; join transactions to profiles; compute sequence deltas and trailing baseline | `transactions.csv`, `fx_rates.csv`, `card_profiles.csv` | featured rows (in-memory or CSV) |
| `score.py` | Per-row score + reason; aggregate to card-level decision | featured rows | scored rows |
| `evaluate.py` | Cost-weighted threshold sweep; per-scenario metrics | scored rows + labels | metrics report |

**Swap contract:** only `score.py` changes when the rule baseline is replaced by
an ML model. `features.py` and `evaluate.py` are untouched. This is what makes
the baseline-vs-model comparison a fair test.

---

## 4. Files and interfaces

These are the foundation the rest of the spec rests on. Code generation and test
generation both derive from them.

### 4.1 `transactions.csv`

One row per transaction. Sorted by `(card_id, timestamp)`.

| Field | Type | Notes |
|---|---|---|
| `txn_id` | string | Unique. |
| `card_id` | string | Join key to `card_profiles.csv`. |
| `timestamp` | ISO 8601 | Sequence ordering key within a card. |
| `amount` | float | Positive. **Native currency** (see `currency`), not USD. |
| `currency` | enum (ISO-4217) | Currency of `amount`. Join key to `fx_rates.csv`. |
| `merchant_id` | string | — |
| `merchant_category` | enum | One of the reference categories. |
| `merchant_country` | enum (ISO-2) | — |
| `device_id` | string | New device is a signal. |
| `ip_country` | enum (ISO-2) | Country change is a signal. |
| `entry_mode` | enum | `chip`, `contactless`, `online`, `manual`. |
| `is_fraud` | 0/1 | Ground-truth label, assigned causally by the injector. |
| `scenario` | enum | Provenance label; see §5. Drives per-scenario evaluation. |

**Currency invariant:** `amount` is denominated in `currency`, which is native to
the transaction. Amounts are **not** comparable across rows until normalised to a
common unit (USD) via the `fx_rates.csv` join (§4.2). Every downstream amount
statistic — profiles, baselines, the amount-spike rule — operates on the
normalised USD value, never on raw `amount`. A card may legitimately transact in
more than one currency (travel); foreign currency is therefore **not** a fraud
signal in itself (see §5).

**Label invariant:** `is_fraud` is set by the injection process, never sampled
independently of the features. Within the `impossible_travel` sequence the local
anchor row is genuinely legitimate (`is_fraud=0`); this is realism, not a
labelling defect.

### 4.2 `card_profiles.csv`

One row per card. Sorted by `card_id`.

| Field | Type | Notes |
|---|---|---|
| `card_id` | string | Join key. Primary key here. |
| `n_txns` | int | Transactions observed for this card. |
| `amount_max` | float | — |
| `amount_mean` | float | — |
| `amount_median` | float | Robust centre; preferred over mean for ratio signals. |
| `distinct_countries` | int | — |
| `distinct_devices` | int | — |

**Baseline policy (decision):** the static profile is computed from **all** rows,
not legitimate rows only. Rationale: in production, labels do not exist at
profile-build time, so an all-rows baseline is the honest reflection of what a
real system knows. The known cost — fraud rows nudge the amount statistics upward
and slightly mask the anomaly — is accepted here and recovered by the trailing
baseline in §6.

**Currency policy (decision):** all amount statistics here (`amount_max`,
`amount_mean`, `amount_median`) are computed on **USD-normalised** amounts, not
native `amount`. `profile.py` performs the `fx_rates.csv` join itself rather than
depending on a pre-normalised input, keeping the module independently runnable
(§9). The conversion is a single shared function (`fx.to_usd`) imported by both
`profile.py` and `features.py`, so the rule lives in one place despite the join
appearing in two modules.

### 4.3 Featured row (output of `features.py`)

Transaction fields plus joined profile fields plus derived signals (§6). Carries
`is_fraud` and `scenario` through untouched for evaluation. Amount-based signals
use the USD-normalised amount (§4.4).

### 4.4 `fx_rates.csv`

One row per currency. Static — no time dimension.

| Field | Type | Notes |
|---|---|---|
| `currency` | enum (ISO-4217) | Primary key. Join key from `transactions.csv`. |
| `rate_to_usd` | float | Multiply native `amount` by this to get USD. USD itself has rate `1.0`. |

**Normalisation rule:** `amount_usd = amount * rate_to_usd`. Implemented once in a
shared `fx.to_usd(amount, currency, rates)` helper, imported by both `profile.py`
and `features.py` (§4.2 currency policy).

**Scope simplification (decision):** rates are static — a single rate per
currency, no date dimension. A real system has time-varying rates, and a
transaction would be converted at the rate prevailing on its timestamp. That is
deliberately out of scope here; flagged so the simplification is visible rather
than accidental. The `fx_rates.csv` interface could gain a `date` column and a
point-in-time lookup without disturbing the rest of the pipeline.

**Join invariant:** every `currency` value appearing in `transactions.csv` must
have exactly one row in `fx_rates.csv` (the FX join is lossless; no transaction
can fail to normalise).

---

## 5. Fraud scenarios and hard negatives

Each fraud scenario leaves a distinct fingerprint; each hard negative is its
near-twin, separated by exactly one dimension. The hard negatives are the reason
a single-row threshold model fails on precision — by construction, the only thing
distinguishing fraud from its twin is a *sequence* property.

| Fraud scenario | Fingerprint | Hard-negative twin | Separating dimension |
|---|---|---|---|
| `card_testing` | Many tiny online amounts, minutes apart, new device + merchant | — | velocity / amount cluster |
| `account_takeover` | New device + new IP country, escalating amounts | `hard_neg_new_device`: new device, **home** country, normal amounts | country + amount |
| `impossible_travel` | Legit local txn, then far-country txn minutes later | `hard_neg_travel`: far country, but **hours** apart | time delta |
| `stolen_spree` | Run of mid/large purchases, unusual categories | `hard_neg_big_ticket`: **one** large legit purchase | run length |

**Requirement:** the dataset must contain all four fraud scenarios and all three
hard-negative types, at configurable injection rates, with a realistic class
imbalance (default fraud rate well below 1% of rows).

**Currency neutrality:** fraud may occur in any currency, and a foreign-currency
transaction is **not** a fraud signal in itself — legitimate travel produces
foreign-currency transactions too. The generator must not make currency a
giveaway (e.g. by transacting fraud only in a distinctive currency), or the hard
negatives lose their force. Currency is realism, not a label proxy.

---

## 6. Feature requirements (`features.py`)

For each transaction, joined to its card profile, compute at minimum:

- **USD normalisation (first step):** convert native `amount` to `amount_usd` via
  the `fx_rates.csv` join (`fx.to_usd`). All amount-based signals below use
  `amount_usd`, never native `amount`.
- **Amount-vs-baseline ratio:** `amount_usd / amount_median` from the profile
  (profile median is itself USD, §4.2).
- **Sequence deltas** (per card, ordered by timestamp):
  - seconds since previous transaction;
  - rolling transaction count in a trailing time window (velocity);
  - is-new-device (device not seen earlier for this card);
  - is-country-change (merchant/IP country differs from previous);
  - is-new-merchant.
- **Trailing point-in-time baseline:** running median/mean of `amount_usd` over
  the card's *prior* transactions only. This recovers the signal the all-rows
  static profile masks, with no look-ahead. Where static and trailing baselines
  diverge is itself reportable.

**Leakage requirement:** no derived feature may use information from a
transaction's own future. The static profile join (§4.2) and the FX-rate join
(§4.4) are the permitted exceptions — both are static lookups, not future
transaction data — and are explicitly flagged as such; the trailing baseline is
the production-correct counterpart and both may be carried for comparison.

---

## 7. Scoring requirements (`score.py`)

- **Per-row score** in `[0, 1]` plus a **reason string** naming which rule fired
  and why. The reason string is a hard requirement, not a nicety: it is what
  makes the baseline explainable and is reused in stakeholder and README
  narrative.
- **v1 scorer is rule-based and transparent.** Rules map directly to the
  fingerprints in §5 (velocity burst, amount escalation, country change without
  travel time, run of large unusual-category purchases).
- **Card-level aggregation** of row scores to an account decision. The
  aggregation function is an explicit design choice and must be named, not
  implicit: `max` row-score is the simplest; a decaying sum better matches
  many-small-signal attacks (card testing) versus a single large signal. The
  chosen function is configurable.
- **Pluggability:** the scorer exposes a stable interface so an ML model can
  replace the rules without touching `features.py` or `evaluate.py`.

---

## 8. Evaluation requirements (`evaluate.py`)

The centrepiece. Takes scored rows plus ground-truth labels and produces:

- **Threshold sweep:** confusion counts (TP/FP/TN/FN) across the decision
  threshold range.
- **Two configurable cost models, as knobs:**
  1. **Fixed ratio** — a false negative costs N times a false positive
     (e.g. 1:100). Captures the asymmetry without dollar figures.
  2. **Amount-weighted** — false-negative cost is a function of the dollars at
     risk; false-positive cost is a per-review operating cost. The exact
     false-negative cost function (full amount lost / capped at issuer liability /
     flat per-miss) is a configurable policy and an open decision (§11).
- **Cost-minimising threshold** under each cost model.
- **Per-scenario recall breakdown** — recall reported separately for
  `card_testing`, `account_takeover`, `impossible_travel`, `stolen_spree`. A
  single blended number hides that a detector may catch sprees and miss card
  testing entirely. This breakdown is a required output, not optional.
- **Hard-negative false-positive rate** — reported against a naive single-row
  threshold baseline, to demonstrate the sequence approach earns its complexity.

**Metric requirement:** report precision, recall, and total cost. Do **not**
report accuracy as a headline metric; at the project's class imbalance it is
actively misleading and the brief should say so.

---

## 9. Non-functional requirements

- **Reproducibility:** all randomness seeded; same seed yields identical output.
- **No external services:** pure file-based I/O (CSV), no database, no network.
  Runnable from a clean checkout with the standard library plus a minimal,
  declared dependency set.
- **Determinism for tests:** every module callable as a pure function on
  in-memory data, independent of file I/O, so acceptance tests need no fixtures
  on disk.
- **Readability over cleverness:** the code is a portfolio and discussion
  artifact; clarity is a requirement, not a preference.

---

## 10. Acceptance criteria (test-generation source)

Each criterion below is directly translatable to a test case.

**Generator**
- G1. With a fixed seed, two runs produce byte-identical `transactions.csv`.
- G2. Every `is_fraud=1` row has a non-`legit`, non-`hard_neg_*` scenario; every
  `hard_neg_*` row has `is_fraud=0`.
- G3. All four fraud scenarios and all three hard-negative types appear at
  default rates on a sufficiently large run.
- G4. Output is sorted by `(card_id, timestamp)`.
- G5. Every row has a valid ISO-4217 `currency`, and every currency appearing in
  `transactions.csv` has exactly one row in `fx_rates.csv` (lossless FX join).
- G6. Currency is not a label proxy: the distribution of `currency` over
  `is_fraud=1` rows is not materially different from its distribution over
  legitimate rows (currency does not leak the label).

**FX rates**
- X1. `fx_rates.csv` has exactly one row per currency; USD is present with
  `rate_to_usd == 1.0`; all rates are positive.

**Profile**
- P1. Exactly one profile row per distinct `card_id` in the input.
- P2. `amount_max >= amount_mean >= 0` and `amount_median >= 0` for every card.
- P3. `distinct_countries` and `distinct_devices` are ≥ 1 and ≤ that card's
  transaction count.
- P4. Profiles join losslessly to transactions on `card_id` (no orphans either
  side).
- P5. Profile amount statistics are computed on USD-normalised amounts:
  `fx.to_usd` applied before aggregation (a card transacting only in non-USD
  currency has profile amounts differing from its native-amount aggregates by the
  rate).

**Features**
- F1. No feature for a transaction uses any later transaction of the same card
  (leakage check), except the explicitly-flagged static profile and FX joins.
- F2. is-new-device is false for the first transaction's home device and true the
  first time an unseen device appears.
- F3. seconds-since-previous is null/sentinel for a card's first transaction and
  non-negative thereafter.
- F4. `amount_usd == amount * rate_to_usd` for the row's currency, within
  floating-point tolerance; all amount-vs-baseline ratios use `amount_usd`.

**Scoring**
- S1. Every scored row has a score in `[0, 1]` and a non-empty reason string.
- S2. Replacing the scorer implementation produces no change in the
  feature or evaluation module interfaces.
- S3. Card-level aggregation is deterministic for a fixed input and configuration.

**Evaluation**
- E1. The threshold sweep is monotonic in the expected direction (raising the
  threshold cannot increase the flagged count).
- E2. Per-scenario recall is reported for all four fraud scenarios.
- E3. The reported cost-minimising threshold actually minimises total cost over
  the swept range under the active cost model.
- E4. Accuracy does not appear as a headline metric.

---

## 11. Open decisions

- **False-negative cost function** in the amount-weighted model: full transaction
  amount lost, amount capped at issuer liability, or a flat per-miss cost.
  *(Pending; default proposal: amount-weighted with a configurable cap.)*
- **Card-level aggregation default:** `max` row-score versus decaying sum.
  *(Pending; decaying sum favoured for card-testing sensitivity.)*
- **Trailing-baseline window:** count-based (last N txns) versus time-based
  (trailing days). *(Pending.)*

---

## 12. Out of scope / future work

- ML scorer swap-in and its A/B comparison against the rule baseline on this
  harness.
- An agentic investigation layer that consumes high-score accounts and produces a
  written case file (shares the explanation backbone with `score.py`).
- Streaming / online evaluation; this brief covers batch only.
