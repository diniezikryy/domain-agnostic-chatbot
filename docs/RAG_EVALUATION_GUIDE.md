# RAG Evaluation with Industry-Standard Metrics

Complete guide to evaluating your RAG chatbot using the **RAGAS framework** - industry-standard metrics for Retrieval Augmented Generation systems.

---

## Quick Start

### Single Query Comparison

```bash
# With LLM-as-judge (accurate, uses tokens)
python interactive_comparison_standard.py "Your query here" critical_illness true

# With heuristic evaluation (free, fast)
python interactive_comparison_standard.py "Your query here" critical_illness false
```

### Interactive Mode

```bash
python interactive_comparison_standard.py
```

Then enter queries one at a time. Type `quit` to exit.

---

## What This Tool Does

Compares your RAG system against a baseline LLM (without retrieval) using **industry-standard RAGAS metrics**:

### 1. Performance Comparison
- Baseline response time
- RAG response time  
- Overhead calculation

### 2. Side-by-Side Responses
- Baseline LLM (no documents)
- RAG Pipeline (with retrieved documents)
- Visual comparison in table format

### 3. RAGAS Metrics (Industry Standard)
- **Faithfulness**: Factual consistency with documents (0-1)
- **Answer Relevance**: How well answer addresses query (0-1)
- **Context Precision**: Quality of retrieved documents (0-1)
- **Context Recall**: Completeness of retrieval (0-1)
- **RAGAS Score**: Harmonic mean of all 4 metrics

### 4. Performance Analysis
- Dimension breakdown ([+] improved, [=] stable, [-] declined)
- Winner determination (or "Tie" if within 5%)
- Professional assessment

---

## Why RAGAS? (Industry Standards)

### The Problem with Custom Metrics

Previous evaluation used **custom heuristics** that were **not based on standardized benchmarks**:

```python
# Custom heuristics
- Citation count (simple regex patterns)
- Hallucination indicators (word counting)
- Comprehensiveness (structure check)
- Weighted formula: 30% + 30% + 40%
```

**Problems:**
- ❌ No academic backing
- ❌ Not comparable across systems
- ❌ Arbitrary weights
- ❌ Simple pattern matching

### The Solution: RAGAS Framework

Now using **industry-standard metrics** based on:

1. **RAGAS Framework** (Retrieval Augmented Generation Assessment)
2. **Academic research** best practices
3. **Production-proven** evaluation methods

```python
# Industry standards
- Faithfulness (factual consistency)
- Answer Relevance (query alignment)  
- Context Precision (retrieval quality)
- Context Recall (information coverage)
- RAGAS Score (harmonic mean)
```

**Advantages:**
- ✅ Based on academic research
- ✅ Used in production RAG systems
- ✅ Comparable across different RAG implementations
- ✅ Supports both LLM-as-judge and heuristic evaluation
- ✅ Harmonic mean ensures balanced performance

---

## RAGAS Metrics Explained

### Core Metrics

| Metric | Definition | What It Measures |
|--------|-----------|------------------|
| **Faithfulness** | Factual consistency with retrieved context | Does the answer make claims supported by the documents? |
| **Answer Relevance** | How well answer addresses the query | Does the response actually answer what was asked? |
| **Context Precision** | Quality of retrieved documents | Did retrieval surface relevant information? |
| **Context Recall** | Completeness of information retrieval | Was all needed information retrieved? |

### Additional Metrics

- **Hallucination Score**: Inverse of faithfulness (lower = better)
- **Citation Quality**: Proportion of answer supported by explicit citations

---

## Score Interpretation

### RAGAS Score Scale

| Score | Assessment | Meaning |
|-------|-----------|---------|
| 0.9-1.0 | Excellent | Production-ready, high quality |
| 0.7-0.9 | Good | Solid performance, minor improvements possible |
| 0.5-0.7 | Fair | Acceptable but needs optimization |
| 0.3-0.5 | Poor | Significant issues, major improvements needed |
| 0.0-0.3 | Very Poor | System not functioning properly |

### Example Output

```
====================================================================================================
RAGAS Score (Overall)               0.000                0.882                [RAG]
----------------------------------------------------------------------------------------------------
Faithfulness                        0.000                0.778                RAG
Answer Relevance                    0.750                1.000                RAG
Context Precision                   0.000                0.800                RAG
Context Recall                      0.000                1.000                RAG
====================================================================================================

[RESULT] RAG system demonstrates a +0.882 point improvement in RAGAS score
         Assessment: Statistically significant quality enhancement observed

[ANALYSIS] Performance Breakdown by Dimension:
   [+] Faithfulness: +0.778    (RAG answers are factually grounded)
   [+] Relevance: +0.250       (RAG answers the actual question)
   [+] Precision: +0.800       (RAG retrieves relevant docs)
   [+] Recall: +1.000          (RAG retrieves all needed info)
```

**Interpretation:**
- Baseline: RAGAS = 0.000 (no context, all zeros)
- RAG: RAGAS = 0.882 (excellent quality)
- Improvement: +0.882 (significantly better)

---

## Evaluation Methods

### LLM-as-Judge (Recommended)

Uses GPT-4o-mini to evaluate responses with nuanced understanding:

```bash
python interactive_comparison_standard.py "Your query" critical_illness true
```

**Pros:** 
- More accurate
- Understands semantic meaning
- Detects subtle issues

**Cons:** 
- Uses API tokens (~$0.0001 per evaluation)

**When to Use:**
- Final evaluations
- Official comparisons
- Production readiness checks

### Heuristic (Fast & Free)

Uses text overlap and pattern matching:

```bash
python interactive_comparison_standard.py "Your query" critical_illness false
```

**Pros:** 
- Free
- Fast
- Deterministic

**Cons:** 
- Less accurate
- Misses nuanced cases

**When to Use:**
- Quick testing
- Development iteration
- Rapid prototyping

---

## How Metrics Are Calculated

### Faithfulness

**LLM-Judge:**
- Prompt asks GPT-4o-mini to identify supported vs unsupported claims
- Returns score 0-1 based on proportion of supported claims

**Heuristic:**
- Splits answer into sentences
- Checks if key terms from each sentence appear in retrieved contexts
- Score = proportion of supported sentences

### Answer Relevance

**LLM-Judge:**
- Evaluates if answer addresses all aspects of the query
- Identifies missing aspects

**Heuristic:**
- Extracts key terms from query
- Counts how many appear in answer
- Score = proportion of query terms in answer

### Context Precision

**LLM-Judge:**
- Evaluates each retrieved context for relevance to query
- Score = proportion of relevant contexts

**Heuristic:**
- Checks term overlap between query and each context
- Context is "relevant" if >30% of query terms present

### Context Recall

**Without Ground Truth:**
- Checks if contexts support the facts mentioned in answer
- Score = proportion of answer facts found in contexts

**With Ground Truth:**
- Compares retrieved contexts to known correct context
- Score = proportion of ground truth information retrieved

### RAGAS Score (Harmonic Mean)

```python
RAGAS = 4 / (1/faithfulness + 1/relevance + 1/precision + 1/recall)
```

**Why harmonic mean?**
- More sensitive to low scores than arithmetic mean
- Ensures balanced performance (can't game one metric)
- Standard in information retrieval research

---

## Usage Examples

### Quick Test (Free)

```bash
python interactive_comparison_standard.py "What are the exclusions?" critical_illness false
```

### Accurate Evaluation (Uses Tokens)

```bash
python interactive_comparison_standard.py "What is the waiting period?" critical_illness true
```

### Interactive Testing Session

```bash
python interactive_comparison_standard.py
# Then enter: critical_illness
# Then enter: Y (for LLM-judge)
# Then enter queries one by one
```

---

## Requirements

- OpenAI API key in `.env`
- At least one batch setup (run `setup_batch.py` first)
- Dependencies installed (`pip install -r requirements.txt`)

---

## Academic References

### Papers & Frameworks

1. **RAGAS Framework**: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)
2. **Retrieval Metrics**: nDCG, MAP, MRR from information retrieval research
3. **Faithfulness Evaluation**: Factual consistency research from NLP community

### Production Frameworks

- **RAGAS** (Python library)
- **TruLens** (Explainable RAG evaluation)
- **DeepEval** (Open-source evaluation)
- **Arize Phoenix** (Production monitoring)

### Standard Benchmarks

- **Precision@k, Recall@k**: Standard retrieval metrics since 1960s
- **Faithfulness**: Adapted from fact-checking and NLI research  
- **Answer Relevance**: Standard in QA system evaluation

---

## Why This Matters

### Before (Custom Metrics)

```
"My RAG scored 0.715 vs baseline 0.400"
→ What does this mean?
→ How does it compare to other RAG systems?
→ Is 0.715 good or bad?
→ Can't compare to published results
```

### After (Industry Standards)

```
"My RAG scored RAGAS 0.882"
→ This is EXCELLENT (0.8+ is production-grade)
→ Comparable to published RAG benchmarks
→ Faithfulness 0.778 = most claims are grounded
→ Can compare to other systems using RAGAS
```

---

## Next Steps

1. **Run full evaluation** with all ground truth queries using standard metrics
2. **Compare RAGAS scores** to determine production readiness
3. **Optimize based on dimension breakdown** (which metric is lowest?)
4. **Benchmark against published results** for your domain

Run this to evaluate your full query set:

```bash
python interactive_comparison_standard.py
# Then enter all your queries one by one
```

---

## Key Takeaways

1. ✅ **Industry-standard RAGAS framework** - academically backed
2. ✅ **Two evaluation modes** - LLM-judge (accurate) or heuristic (fast)
3. ✅ **Comparable results** - benchmark against published RAG systems
4. ✅ **Harmonic mean** - ensures balanced quality across all dimensions
5. ✅ **Production-proven** - used by leading RAG implementations

---

**Implementation Files:**
- Metrics: `evaluation/industry_standard_evaluator.py`
- Interactive Tool: `interactive_comparison_standard.py`
