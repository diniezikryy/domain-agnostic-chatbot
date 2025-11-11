---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

# **RAGAS Evaluation and Improvement Plan**

Project: domain-agnostic-chatbot-ui  
Objective: To implement a robust, industry-standard evaluation harness using the RAGAS library. This plan will guide us through a "flywheel" process: Prepare \-\> Run \-\> Evaluate \-\> Improve, enabling us to scientifically measure and enhance our pipeline's performance.

## **The Core Challenge: Refactoring for Testability**

Your current query\_processor.py is designed for a streaming UI, merging retrieval and generation into a single, monolithic function (process\_query\_stream).

**RAGAS cannot evaluate this.**

RAGAS metrics like context\_precision and faithfulness require a "seam" between retrieval and generation. We must be able to:

1. **Run Retrieval:** Get the contexts (the chunks) your pipeline finds.  
2. **Run Generation:** Get the answer your LLM produces *from those specific contexts*.

Therefore, our first and most critical phase is refactoring.

## **Phase 1: Prepare \- Refactoring for Testability**

We will modify query\_processor.py to add two new, non-streaming, testable functions: run\_retrieval and run\_generation.

### **Action 1.1: Modify query\_processor.py**

\# In diniezikryy/domain-agnostic-chatbot/domain-agnostic-chatbot-ui/query\_processor.py  
import json  
import os  
import time  
from pathlib import Path  
from typing import List, Dict, Any, Optional

from openai import OpenAI  
from batch\_manager import BatchManager  
from utils.search import HybridSearchEngine  
from research import DeepResearch \# Your web research module

class QueryProcessor:  
    \# ... (all existing \_\_init\_\_, \_ensure\_batch\_loaded, \_deduplicate\_results, etc. stay) ...  
    \# ... (all \_expand\_query, \_get\_intent\_prompt, \_format\_rag\_context, etc. stay) ...

    \# \=========================================================================  
    \# \== NEW FUNCTION 1: TESTABLE RETRIEVAL  
    \# \=========================================================================  
    def run\_retrieval(self, query: str, batch\_id: str, user\_profile: Optional\[Dict\]) \-\> Dict\[str, Any\]:  
        """  
        Runs the full retrieval pipeline (intent, expansion, RAG, and web research).  
        Returns a dictionary containing all retrieved contexts.  
        """  
        print(f"--- \[EVAL\] Running Retrieval for Query: {query} \---")  
          
        \# 1\. Analyze Intent  
        try:  
            intent\_response \= self.client.chat.completions.create(  
                model="gpt-4o",  
                messages=\[{"role": "user", "content": self.\_get\_intent\_prompt(query)}\],  
                response\_format={"type": "json\_object"},  
                temperature=0.1  
            )  
            intent \= json.loads(intent\_response.choices\[0\].message.content)  
        except Exception as e:  
            print(f"Error analyzing intent: {e}")  
            intent \= {"needs\_comparison": False, "asks\_about\_uncovered\_features": False, "requires\_external\_info": False}

        \# 2\. Expand and Search Documents (RAG)  
        expanded\_query \= self.\_expand\_query(query)  
        raw\_search\_results \= self.search\_engine.hybrid\_search(  
            query=expanded\_query, top\_k=50 \# Retrieve a large candidate set  
        )  
        unique\_results \= self.\_deduplicate\_results(raw\_search\_results)  
          
        \# 3\. Determine if Web Research is Needed  
        \# NOTE: We keep your existing (flawed) logic here to test it  
        needs\_research \= (  
            intent\["needs\_comparison"\] or  
            intent\["asks\_about\_uncovered\_features"\] or  
            intent\["requires\_external\_info"\] or  
            len(unique\_results) \== 0  
        )

        \# 4\. Run Web Research if Needed  
        research\_results \= {"answer": "", "sources": \[\]}  
        if needs\_research:  
            print("--- \[EVAL\] Triggering DeepResearch (Web Search) \---")  
            researcher \= DeepResearch()  
            enhanced\_query \= self.\_format\_enhanced\_query(query, unique\_results, intent)  
            research\_results \= researcher.research(enhanced\_query)  
          
        \# 5\. Collate all contexts for RAGAS  
        rag\_contexts \= \[chunk.get("content", "") for chunk in unique\_results\]  
        web\_contexts \= \[research\_results\["answer"\]\] if research\_results.get("answer") else \[\]

        return {  
            "rag\_chunks\_details": unique\_results, \# For generation  
            "rag\_contexts\_list": rag\_contexts,     \# For RAGAS  
            "web\_contexts\_list": web\_contexts,     \# For RAGAS  
            "web\_research\_raw": research\_results   \# For generation  
        }

    \# \=========================================================================  
    \# \== NEW FUNCTION 2: TESTABLE GENERATION (Non-Streaming)  
    \# \=========================================================================  
    def run\_generation(self, query: str, rag\_chunks: List\[Dict\], research\_results: Dict, user\_profile: Optional\[Dict\]) \-\> str:  
        """  
        Runs the generation step given a query and retrieved contexts.  
        Returns a single, complete answer string.  
        (This is a non-streaming version of your \_generate\_response\_stream)  
        """  
        print(f"--- \[EVAL\] Running Generation for Query: {query} \---")  
          
        \# 1\. Format Document Context  
        context\_from\_docs \= self.\_format\_rag\_context\_for\_prompt(rag\_chunks)

        \# 2\. Format Profile Context  
        profile\_info, salutation \= self.\_format\_profile\_for\_prompt(user\_profile)

        \# 3\. Combine Web Research into the prompt  
        research\_context \= ""  
        if research\_results and research\_results.get("answer"):  
            research\_context \= f"\\n--- EXTERNAL WEB RESEARCH \---\\n{research\_results\['answer'\]}\\n--- END OF WEB RESEARCH \---"

        \# 4\. Create the Final Prompt (This is your key prompt)  
        prompt\_instructions \= f"""  
        {salutation}  
        Your task is to answer the user's question using the provided documents AND external research.

        User Question: {query}  
        {profile\_info}

        \--- POLICY DOCUMENT CHUNKS \---  
        {context\_from\_docs if context\_from\_docs else "No relevant information found in policy documents."}  
        \--- END OF DOCUMENTS \---  
        {research\_context}

        CRITICAL RESPONSE RULES:  
        1\. Base your answer on BOTH document chunks and external research.  
        2\. Prioritize document information if available.  
        3\. Use the user's specific policy tier (e.g., "P PLUS") when citing benefits.  
        4\. Cite document facts with \[Source X: filename.pdf, Page Y\].  
        5\. If using web research, state that (e.g., "External research shows...").  
        """  
          
        \# 5\. Call OpenAI API (non-streaming)  
        try:  
            response \= self.client.chat.completions.create(  
                model="gpt-4-1106-preview", \# Or your preferred model  
                messages=\[  
                    {"role": "system", "content": "You are an expert financial advisor..."},  
                    {"role": "user", "content": prompt\_instructions},  
                \],  
                max\_tokens=1500,  
                temperature=0.0,  
            )  
            final\_answer \= response.choices\[0\].message.content.strip()  
            return final\_answer  
        except Exception as e:  
            return f"Error generating response: {e}"

    \# Helper function to extract generation prompt logic  
    def \_format\_rag\_context\_for\_prompt(self, rag\_chunks: List\[Dict\]) \-\> str:  
        context\_parts \= \[\]  
        max\_chunks \= 30 \# Your original limit  
        for i, result in enumerate(rag\_chunks\[:max\_chunks\], 1):  
            content \= result.get("content", "").strip()  
            metadata \= result.get("metadata", {})  
            if content:  
                filename \= metadata.get("filename", "Unknown")  
                page \= metadata.get("page\_number", "N/A")  
                source\_ref \= f"\[Source {i}: {filename}, Page {page}\]"  
                context\_parts.append(f"{source\_ref}\\n{content}")  
        return "\\n\\n---\\n\\n".join(context\_parts)

    \# Helper function to extract profile prompt logic  
    def \_format\_profile\_for\_prompt(self, user\_profile: Optional\[Dict\]) \-\> tuple\[str, str\]:  
        if user\_profile:  
            user\_name \= user\_profile.get("name", "User")  
            policy\_tiers \= user\_profile.get("policy\_tiers", {})  
            profile\_info \= f"\\n\\nUSER PROFILE:\\n- User Name: {user\_name}"  
            if policy\_tiers:  
                profile\_info \+= "\\n- Policy Tiers:"  
                for policy, tier in policy\_tiers.items():  
                    profile\_info \+= f"\\n  \- {policy}: {tier} plan"  
        else:  
            user\_name \= "User"  
            profile\_info \= ""  
          
        salutation \= f"Hi {user\_name.split()\[0\] if user\_name \!= 'User' else 'Hi'},"  
        return profile\_info, salutation

## **Phase 2: Prepare \- The "Golden Set"**

We will create a test\_data/ directory to hold our test files, based on the user\_profile.json you provided in the repo and the three PDFs.

### **Action 2.1: Create Test Profile**

Copy your existing user\_profile.json into the new test directory.

**test\_data/profile\_1.json:**

{  
  "user\_id": "TEST\_USER\_001",  
  "name": "Dinie",  
  "date\_of\_birth": "2001-08-01",  
  "smoking\_status": "non-smoker",  
  "policy\_tiers": {  
    "GREAT\_SupremeHealth\_Benefits.pdf": "P PLUS",  
    "GREAT\_TravelCare.pdf": "Platinum"  
  },  
  "policies\_owned": \[  
    "Manulife\_Policy\_Illustration\_REDACTED.pdf",  
    "GREAT\_SupremeHealth\_Benefits.pdf",  
    "GREAT\_TravelCare.pdf"  
  \],  
  "policy\_details": \[  
     // (Copied from your user\_profile.json)  
  \]  
}

### **Action 2.2: Create Evaluation Dataset**

This dataset is built *from your PDFs* and targets your pipeline's specific features and flaws.

**test\_data/evaluation\_dataset.json:**

\[  
  {  
    "question\_id": "TEST\_001\_BASELINE\_RAG",  
    "question": "What is the annual benefit limit for GREAT SupremeHealth?",  
    "ground\_truth": "The annual benefit limit for the GREAT SupremeHealth P PLUS plan is S$1,500,000.",  
    "user\_profile\_id": "profile\_1"  
  },  
  {  
    "question\_id": "TEST\_002\_PERSONALIZATION",  
    "question": "What is my co-insurance?",  
    "ground\_truth": "Your co-insurance for your GREAT SupremeHealth (P PLUS plan) is 10% of the claim, after the deductible is met.",  
    "user\_profile\_id": "profile\_1"  
  },  
  {  
    "question\_id": "TEST\_003\_DEEP\_RESEARCH\_FALLBACK",  
    "question": "What are the market alternatives to the Manulife ManuProtect Term (II)?",  
    "ground\_truth": "While your documents do not list alternatives, external web research shows popular alternatives for term life insurance in Singapore include policies from insurers like Singlife, AIA, and NTUC Income, which offer similar death and TPD benefits.",  
    "user\_profile\_id": "profile\_1"  
  },  
  {  
    "question\_id": "TEST\_004\_CHUNK\_FLAW",  
    "question": "What is the benefit for 'Post Hospitalisation Treatment' for the 'P PLUS' plan?",  
    "ground\_truth": "The benefit for Post Hospitalisation Treatment for the P PLUS plan is 'As Charged', covered for 180 days after discharge. (From GREAT\_SupremeHealth\_Benefits.pdf, Page 12)",  
    "user\_profile\_id": "profile\_1"  
  },  
  {  
    "question\_id": "TEST\_005\_COMPARISON\_FLAW",  
    "question": "Compare the medical evacuation coverage in my GREAT TravelCare and my GREAT SupremeHealth policies.",  
    "ground\_truth": "Your GREAT TravelCare policy (Platinum tier) provides S$1,000,000 for Emergency Medical Evacuation. Your GREAT SupremeHealth (P PLUS plan) covers 'Emergency Treatment Overseas' as charged.",  
    "user\_profile\_id": "profile\_1"  
  }  
\]

* **Test 1 (Baseline):** Checks simple RAG.  
* **Test 2 (Personalization):** Checks if the policy\_tiers from profile\_1.json are correctly used.  
* **Test 3 (Web Fallback):** Checks the DeepResearch fallback for a question *not* in the docs.  
* **Test 4 (Chunking Flaw):** Targets a specific row in the GREAT\_SupremeHealth table. A low score here will prove your naive chunker (\_create\_chunks) is breaking the table's context.  
* **Test 5 (Comparison Flaw):** This question is fully answerable by the docs, but your pipeline will *still* run a web search because the *intent* is "comparison." We will use this to measure the inefficiency.

## **Phase 3: Run \- The Evaluation Harness Script**

This script, run\_evaluation.py, will orchestrate the entire evaluation. It is designed to be modular to support the experiments in Phase 5\.

### **Action 3.1: Install Dependencies**

Add ragas, datasets, and flashrank to your requirements.txt and run pip install \-r requirements.txt.

### **Action 3.2: Create run\_evaluation.py**

import json  
import os  
import asyncio  
from pathlib import Path  
from datasets import Dataset  
from ragas import evaluate  
from ragas.metrics import (  
    faithfulness,  
    answer\_relevancy,  
    context\_precision,  
    context\_recall,  
    answer\_correctness  
)  
from dotenv import load\_dotenv  
import pandas as pd  
import argparse \# To select experiments  
from flashrank import Ranker

\# Import your refactored classes  
from query\_processor import QueryProcessor  
from batch\_manager import BatchManager

\# \--- CONFIGURATION \---  
EVAL\_DATASET\_PATH \= "test\_data/evaluation\_dataset.json"  
TEST\_PROFILES\_DIR \= "test\_data"  
\# This MUST match the user\_id/batch\_id you set up in the UI  
\# Register a user, upload the 3 PDFs, then find their batch\_id in the \`batches/\` dir  
DEFAULT\_TEST\_BATCH\_ID \= "user\_1" \# \<-- \*\*\* UPDATE THIS to match your test user's batch ID \*\*\*  
\# \---------------------

def load\_data():  
    """Loads the golden set and test profiles."""  
    with open(EVAL\_DATASET\_PATH, 'r') as f:  
        golden\_set \= json.load(f)  
      
    profiles \= {}  
    for item in golden\_set:  
        profile\_id \= item.get("user\_profile\_id")  
        if profile\_id and profile\_id not in profiles:  
            profile\_path \= Path(TEST\_PROFILES\_DIR) / f"{profile\_id}.json"  
            if profile\_path.exists():  
                with open(profile\_path, 'r') as f:  
                    profiles\[profile\_id\] \= json.load(f)  
    return golden\_set, profiles

\# \--- Experimental Pipeline Functions \---

def run\_retrieval\_baseline(query\_processor, query, batch\_id, user\_profile):  
    """Runs the default retrieval pipeline as you defined it."""  
    return query\_processor.run\_retrieval(query, batch\_id, user\_profile)

def run\_generation\_baseline(query\_processor, query, retrieval\_data, user\_profile):  
    """Runs the default generation pipeline."""  
    return query\_processor.run\_generation(  
        query=query,  
        rag\_chunks=retrieval\_data\["rag\_chunks\_details"\],  
        research\_results=retrieval\_data\["web\_research\_raw"\],  
        user\_profile=user\_profile  
    )

def run\_generation\_no\_rag(query\_processor, query, retrieval\_data, user\_profile):  
    """Experiment 1: Runs generation with NO RAG context."""  
    print("--- \[EVAL\] Running Generation (NO\_RAG) \---")  
    \# We pass empty contexts  
    return query\_processor.run\_generation(  
        query=query,  
        rag\_chunks=\[\], \# Empty  
        research\_results={}, \# Empty  
        user\_profile=user\_profile  
    )

def run\_retrieval\_rerank(query\_processor, query, batch\_id, user\_profile):  
    """Experiment 2: Runs retrieval and adds a re-ranking step."""  
    print("--- \[EVAL\] Running Retrieval (RE-RANKING) \---")  
    retrieval\_data \= query\_processor.run\_retrieval(query, batch\_id, user\_profile)  
      
    \# Initialize re-ranker (cached)  
    if not hasattr(query\_processor, 'reranker'):  
        query\_processor.reranker \= Ranker(model\_name="ms-marco-MiniLM-L-12-v2")

    \# Only re-rank the document chunks  
    rag\_chunks \= retrieval\_data\["rag\_chunks\_details"\]  
    if rag\_chunks:  
        passages \= \[{"id": i, "text": chunk\["content"\]} for i, chunk in enumerate(rag\_chunks)\]  
        reranked \= query\_processor.reranker.rerank(query=query, passages=passages)  
          
        \# Get the top 5 reranked chunks  
        reranked\_indices \= {r\['id'\] for r in reranked\[:5\]}  
        final\_rag\_chunks \= \[c for i, c in enumerate(rag\_chunks) if i in reranked\_indices\]  
          
        \# Update the retrieval data with the re-ranked chunks  
        retrieval\_data\["rag\_chunks\_details"\] \= final\_rag\_chunks  
        retrieval\_data\["rag\_contexts\_list"\] \= \[c\["content"\] for c in final\_rag\_chunks\]  
        print(f"Re-ranked from {len(rag\_chunks)} to {len(final\_rag\_chunks)} chunks.")

    return retrieval\_data

def run\_retrieval\_hyde(query\_processor, query, batch\_id, user\_profile):  
    """Experiment 3: Runs retrieval using a HyDE query for embeddings."""  
    print("--- \[EVAL\] Running Retrieval (HyDE) \---")  
      
    \# 1\. Generate Hypothetical Answer  
    hyde\_prompt \= f"Write a short, hypothetical answer to the following question. Do not say you don't know. \\nQuestion: {query}"  
    hyde\_response \= query\_processor.client.chat.completions.create(  
        model="gpt-4o-mini",  
        messages=\[{"role": "user", "content": hyde\_prompt}\],  
        max\_tokens=150,  
        temperature=0.0  
    )  
    hyde\_answer \= hyde\_response.choices\[0\].message.content  
    print(f"HyDE Answer: {hyde\_answer\[:100\]}...")

    \# 2\. Run retrieval, but use the HyDE answer for the \*semantic\* part  
    \# We still use the \*original\* query for keyword search (BM25)  
    \# This requires modifying HybridSearchEngine, but for a fast test,  
    \# we can just use the HyDE answer as the \*entire\* query.  
    \# A more advanced version would separate them.  
      
    \# Simple HyDE: Use HyDE answer for BOTH semantic and keyword  
    retrieval\_data \= query\_processor.run\_retrieval(hyde\_answer, batch\_id, user\_profile)  
    return retrieval\_data

\# \--- Main Evaluation Orchestrator \---

def run\_pipeline(questions\_data, profiles\_data, experiment\_name, test\_batch\_id):  
    """  
    Runs the RAG pipeline for all test questions and collects results  
    based on the specified experiment.  
    """  
    print(f"Initializing RAG pipeline for experiment: {experiment\_name}")  
    batch\_manager \= BatchManager()  
    query\_processor \= QueryProcessor(batch\_manager)

    if not query\_processor.\_ensure\_batch\_loaded(test\_batch\_id):  
        raise Exception(f"FATAL: Could not load batch '{test\_batch\_id}'. Did you upload the 3 PDFs to this user?")

    print(f"Successfully loaded test batch '{test\_batch\_id}'.")  
      
    \# Select the functions to run based on the experiment  
    if experiment\_name \== "baseline":  
        retrieval\_func \= run\_retrieval\_baseline  
        generation\_func \= run\_generation\_baseline  
    elif experiment\_name \== "no\_rag":  
        retrieval\_func \= run\_retrieval\_baseline \# Still need to run to get question  
        generation\_func \= run\_generation\_no\_rag  
    elif experiment\_name \== "reranking":  
        retrieval\_func \= run\_retrieval\_rerank  
        generation\_func \= run\_generation\_baseline  
    elif experiment\_name \== "hyde":  
        retrieval\_func \= run\_retrieval\_hyde  
        generation\_func \= run\_generation\_baseline  
    else:  
        raise ValueError(f"Unknown experiment: {experiment\_name}")

    results \= \[\]  
      
    for item in questions\_data:  
        question \= item\["question"\]  
        ground\_truth \= item\["ground\_truth"\]  
        profile\_id \= item\["user\_profile\_id"\]  
        user\_profile \= profiles\_data.get(profile\_id, {})  
          
        print(f"\\n--- Processing Question: {item\['question\_id'\]} \---")  
          
        \# 1\. Run Retrieval  
        retrieval\_data \= retrieval\_func(query\_processor, question, test\_batch\_id, user\_profile)  
          
        \# Collate all contexts for RAGAS  
        all\_contexts \= retrieval\_data\["rag\_contexts\_list"\] \+ retrieval\_data\["web\_contexts\_list"\]  
              
        if not all\_contexts:  
            print("Warning: No context was retrieved.")  
            all\_contexts \= \[\]   
          
        \# For No-RAG, we manually clear contexts for the generator  
        if experiment\_name \== "no\_rag":  
            all\_contexts \= \[\] \# RAGAS will evaluate this as 0 context  
            retrieval\_data\["rag\_chunks\_details"\] \= \[\]  
            retrieval\_data\["web\_research\_raw"\] \= {}

        \# 2\. Run Generation  
        generated\_answer \= generation\_func(query\_processor, question, retrieval\_data, user\_profile)  
          
        \# 3\. Store results for RAGAS  
        results.append({  
            "question": question,  
            "answer": generated\_answer,  
            "contexts": all\_contexts,  
            "ground\_truth": ground\_truth,  
        })  
          
        print(f"Generated Answer: {generated\_answer\[:100\]}...")  
        print(f"Retrieved {len(all\_contexts)} context chunks.")

    return results

def main():  
    parser \= argparse.ArgumentParser(description="Run RAGAS Evaluation")  
    parser.add\_argument(  
        "--experiment",  
        type=str,  
        default="baseline",  
        choices=\["baseline", "no\_rag", "reranking", "hyde"\],  
        help="The experiment to run."  
    )  
    parser.add\_argument(  
        "--batch\_id",  
        type=str,  
        default=DEFAULT\_TEST\_BATCH\_ID,  
        help="The test batch ID to use (e.g., user\_1\_small, user\_1\_large)"  
    )  
    args \= parser.parse\_args()

    load\_dotenv()  
    if not os.getenv("OPENAI\_API\_KEY") or not os.getenv("TAVILY\_API\_KEY"):  
        raise ValueError("OPENAI\_API\_KEY and TAVILY\_API\_KEY must be in .env")

    print(f"Starting RAGAS Evaluation for experiment: {args.experiment} on batch: {args.batch\_id}")  
      
    questions, profiles \= load\_data()  
    pipeline\_results \= run\_pipeline(questions, profiles, args.experiment, args.batch\_id)  
    dataset \= Dataset.from\_list(pipeline\_results)  
      
    metrics \= \[  
        faithfulness,  
        answer\_relevancy,  
        context\_precision,  
        context\_recall,  
        answer\_correctness,  
    \]  
      
    print("\\nRunning RAGAS.evaluate()... (This will make LLM calls)")  
    result \= evaluate(dataset, metrics)  
    print("RAGAS Evaluation Complete.")  
      
    df \= result.to\_pandas()  
    print(df.to\_string())  
      
    output\_filename \= f"ragas\_results\_{args.experiment}\_{args.batch\_id}.csv"  
    df.to\_csv(output\_filename, index=False)  
    print(f"\\nResults saved to {output\_filename}")

if \_\_name\_\_ \== "\_\_main\_\_":  
    main()

## **Phase 4: Evaluate & Improve \- The "Flywheel"**

Run python run\_evaluation.py \--experiment=baseline and analyze the output CSV. This is your "EVALUATE" step, which directly informs the "IMPROVE" step.

| Metric | If This Score is LOW... | Diagnosis (Based on your code) | Actionable Improvement Plan |
| :---- | :---- | :---- | :---- |
| **context\_precision** | ...it means your retrieved contexts are full of "noise" (irrelevant chunks). | **No Re-ranker.** Your pipeline retrieves 50 chunks and sends all 30 (max\_chunks\_for\_context) to the LLM. This is the exact "Lost in the Middle" problem from LLM Lab 04\. | **Run Experiment 2: Re-ranking**. If the score improves, integrate the re-ranker into your run\_retrieval\_baseline function. |
| **context\_recall** | ...it means your retriever FAILED to find the *necessary* chunks to answer the question. | **Naive Chunking Flaw.** This score will be low for TEST\_004\_CHUNK\_FLAW. Your \_create\_chunks function in utils/file\_handlers.py is a fixed-size splitter that breaks your markdown tables. | **Implement Semantic Chunking:** In utils/file\_handlers.py, replace \_create\_chunks with a langchain.text\_splitter.MarkdownHeaderTextSplitter. This will use the \# and \#\# headers (from your pymupdf4llm parser) to create semantically meaningful chunks and keep tables intact. |
| **faithfulness** | ...it means the answer is NOT supported by the contexts. This is **hallucination**. | **Symptom of Noisy Context.** This is a *symptom* of low context\_precision. The LLM is given 30 noisy chunks and is forced to invent an answer. | **Fix Re-ranking First.** By fixing context\_precision (giving the LLM 5 *good* chunks instead of 30 *noisy* ones), faithfulness will increase. |
| **answer\_correctness** | ...it means the answer is factually wrong compared to the ground\_truth. | **Symptom of Failed Retrieval.** This is a *symptom* of low context\_recall. The LLM can't be correct if it never received the correct chunk. | **Fix Chunking First.** By fixing context\_recall (ensuring the correct table row is retrieved), answer\_correctness will increase. |

## **Phase 5: Advanced Experiments (The Full Plan)**

Now you can run all your experiments and compare the CSV outputs.

### **Experiment 1: No-RAG (LLM-Only) Baseline**

* **Goal:** Prove the value of RAG.  
* **Command:** python run\_evaluation.py \--experiment=no\_rag  
* **Hypothesis:** answer\_correctness will be near 0.0, as the LLM has no context.

### **Experiment 2: Re-ranking (Industry Best-Practice)**

* **Goal:** Fix context\_precision and faithfulness.  
* **Command:** python run\_evaluation.py \--experiment=reranking  
* **Hypothesis:** context\_precision and faithfulness will significantly increase.

### **Experiment 3: Query Transformation (HyDE)**

* **Goal:** Fix context\_recall for complex questions.  
* **Command:** python run\_evaluation.py \--experiment=hyde  
* **Hypothesis:** context\_recall will improve for complex questions like TEST\_004\_CHUNK\_FLAW.

### **Experiment 4: Embedding Model (Small vs. Large)**

* **Goal:** Test cost vs. performance (your refined request).  
* **How:**  
  1. **Modify document\_processor.py:** Make create\_batch accept embedding\_model\_name and embedding\_dimension arguments. Pass these to EmbeddingGenerator and SearchIndexBuilder.  
  2. **Create a new script create\_experiment\_batches.py:**  
     \# In create\_experiment\_batches.py  
     from document\_processor import DocumentProcessor

     doc\_processor \= DocumentProcessor()  
     doc\_paths \= \[  
         "documents/user\_1/GREAT\_SupremeHealth\_Benefits.pdf",  
         "documents/user\_1/GREAT\_TravelCare.pdf",  
         "documents/user\_1/Manulife\_Policy\_Illustration\_REDACTED.pdf"  
     \] \# Assuming user\_1 has the files

     print("Creating SMALL batch...")  
     doc\_processor.create\_batch(  
         batch\_id="user\_1\_small",  
         document\_paths=doc\_paths,  
         embedding\_model\_name="text-embedding-3-small",  
         embedding\_dimension=1536  
     )

     print("Creating LARGE batch...")  
     doc\_processor.create\_batch(  
         batch\_id="user\_1\_large",  
         document\_paths=doc\_paths,  
         embedding\_model\_name="text-embedding-3-large",  
         embedding\_dimension=3072  
     )

  3. **Run the new script:** python create\_experiment\_batches.py  
  4. **Run evaluation on both batches:**  
     python run\_evaluation.py \--experiment=baseline \--batch\_id=user\_1\_small  
     python run\_evaluation.py \--experiment=baseline \--batch\_id=user\_1\_large

  5. **Compare** ragas\_results\_baseline\_user\_1\_small.csv and ragas\_results\_baseline\_user\_1\_large.csv.  
* **Hypothesis:** The "large" batch will show a minor improvement in context\_recall for a significant increase in cost and indexing time.