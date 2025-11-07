# **RAG Pipeline Experimentation Plan (for GitHub Copilot)**

## **Overall Main Idea**

The primary goal of this plan is to refactor the existing RAG chatbot into a modular, experiment-ready framework. By parameterizing key components—such as the embedding model, retrieval strategy (Hybrid vs. Vector-Only), query transformation (HyDE), and the generation model—we can systematically run experiments.

The end goal is to use the evaluation/run\_experiments.py script to automatically test each pipeline variation against our full ground-truth dataset (ground\_truth.json \+ custom\_queries.json).

This will produce a comprehensive report with:

1. **RAGAS Metrics:** Faithfulness, Relevance, Context Precision, Context Recall.  
2. **Trustworthiness Metrics:** Hallucination Rate, Accuracy Score.  
3. **Operational Metrics:** Per-Query Latency (seconds) and API Cost (generation tokens).

This allows us to make data-driven decisions on the most optimal RAG configuration, balancing quality, cost, and speed.

## **Stage 0: Parameterize Core Components**

The first step is to make the core components configurable.

### **1\. Modify query\_processor.py**

Update QueryProcessor.\_\_init\_\_ to accept parameters.

**File:** query\_processor.py

**Change this:**

class QueryProcessor:  
    def \_\_init\_\_(self, batch\_manager: BatchManager):  
        self.batch\_manager \= batch\_manager  
        self.search\_engine \= None  
        self.current\_batch\_id \= None  
        \# ... prompts ...

**To this:**

from sentence\_transformers import CrossEncoder \# Add this import  
import numpy as np \# Add this import if not present

class QueryProcessor:  
    def \_\_init\_\_(self,   
                 batch\_manager: BatchManager,   
                 generation\_model: str \= "gpt-4o-mini",  
                 retrieval\_strategy: str \= "hybrid",  
                 use\_hyde: bool \= False,  
                 use\_reranking: bool \= False,  
                 top\_k: int \= 10  
                ):  
        self.batch\_manager \= batch\_manager  
        self.search\_engine \= None  
        self.current\_batch\_id \= None  
          
        \# Experiment parameters  
        self.generation\_model \= generation\_model  
        self.retrieval\_strategy \= retrieval\_strategy  
        self.use\_hyde \= use\_hyde  
        self.use\_reranking \= use\_reranking  
        self.top\_k \= top\_k   
        self.reranker \= None \# For lazy-loading the reranker model  
          
        \# ... prompts ...

### **2\. Modify utils/embeddings.py**

This file is already parameterized to accept model\_name, which is perfect.

**File:** utils/embeddings.py

**Verify this method exists and is correct:**

    def get\_embedding\_dimension(self) \-\> int:  
        """Get the dimension of embeddings for this model."""  
        \# text-embedding-3-small has 1536 dimensions  
        if "text-embedding-3-small" in self.model\_name:  
            return 1536  
        elif "text-embedding-3-large" in self.model\_name:  
            return 3072  
        else:  
            \# Default fallback  
            return 1536

*(This is correct and will support Exp 1.)*

## **Stage 1: Implement Latency/Cost Tracking**

We need to modify the query processors to return token usage and measure latency.

### **1\. Modify query\_processor.py**

Update \_generate\_response to return the response string *and* token count.

**File:** query\_processor.py

**Change this section of \_generate\_response:**

            response \= client.chat.completions.create(  
                model="gpt-4o-mini",  
...  
            )

            return response.choices\[0\].message.content.strip()

        except Exception as e:  
            \# Fallback to basic response if OpenAI fails  
            print(f"OpenAI generation failed: {e}")  
            return f"Based on the policy documents, I found relevant information but couldn't generate a complete response. Key details: {context\[:300\]}..."

**To this:**

            response \= client.chat.completions.create(  
                model=self.generation\_model, \# Use the parameterized model  
                messages=\[  
...  
            )  
              
            tokens\_used \= response.usage.total\_tokens if response.usage else 0  
            return response.choices\[0\].message.content.strip(), tokens\_used

        except Exception as e:  
            \# Fallback to basic response if OpenAI fails  
            print(f"OpenAI generation failed: {e}")  
            return f"Based on the policy documents, I found relevant information but couldn't generate a complete response. Key details: {context\[:300\]}...", 0

Now, update process\_query to handle these new return values and return them.

**File:** query\_processor.py

**Change the end of process\_query from:**

            \# Step 3: Generate response using retrieved context  
            response \= self.\_generate\_response(query, search\_results)

            processing\_time \= time.time() \- start\_time  
            print(f"Processing time: {processing\_time:.2f}s")

            return response

        except Exception as e:  
            return f"Error processing query: {e}"

**To this:**

            \# Step 3: Rerank results if enabled  
            top\_contexts \= self.\_rerank\_results(query, search\_results)

            \# Step 4: Generate response using retrieved context  
            response\_content, tokens\_used \= self.\_generate\_response(query, top\_contexts)

            processing\_time \= time.time() \- start\_time  
            print(f"Processing time: {processing\_time:.2f}s, Tokens: {tokens\_used}")

            return response\_content, tokens\_used, processing\_time

        except Exception as e:  
            return f"Error processing query: {e}", 0, 0

### **2\. Modify evaluation/llm\_baseline\_comparator.py**

Update the baseline comparator to also return tokens and latency.

**File:** evaluation/llm\_baseline\_comparator.py

**Change get\_baseline\_response from:**

    def get\_baseline\_response(self, query: str) \-\> dict:  
...  
            return {  
                'response': answer,  
                'time\_seconds': elapsed\_time  
            }  
              
        except Exception as e:  
...  
            return {  
                'response': f"Error generating baseline response: {str(e)}",  
                'time\_seconds': elapsed\_time  
            }

**To this:**

    def get\_baseline\_response(self, query: str) \-\> dict:  
...  
            answer \= response.choices\[0\].message.content  
            tokens\_used \= response.usage.total\_tokens if response.usage else 0  
            elapsed\_time \= time.time() \- start\_time  
              
            return {  
                'response': answer,  
                'time\_seconds': elapsed\_time,  
                'tokens\_used': tokens\_used  
            }  
              
        except Exception as e:  
...  
            return {  
                'response': f"Error generating baseline response: {str(e)}",  
                'time\_seconds': elapsed\_time,  
                'tokens\_used': 0  
            }

## **Stage 2: Implement Retrieval Strategy Experiments (Exp 5 & Trust.)**

Now, use the parameters created in Stage 0\.

### **1\. Implement Vector-Only Retrieval (Exp 5\)**

First, add the new search method to HybridSearchEngine.

**File:** utils/search.py

**Add this new method** inside the HybridSearchEngine class:

    def vector\_only\_search(self, query: str, top\_k: int \= 10\) \-\> List\[Dict\[str, Any\]\]:  
        """Perform FAISS-only semantic search."""  
        if not self.faiss\_index:  
            print("FAISS index not loaded")  
            return \[\]  
          
        try:  
            \# Get FAISS results (semantic search)  
            faiss\_results \= self.\_faiss\_search(query, top\_k)  
              
            \# Format results similarly to hybrid search for consistency  
            for res in faiss\_results:  
                res\['combined\_score'\] \= res\['score'\]  
              
            return faiss\_results

        except Exception as e:  
            print(f"Error in vector-only search: {e}")  
            return \[\]

Next, use this new method in QueryProcessor.

**File:** query\_processor.py

In the process\_query method, modify the search logic to check self.retrieval\_strategy.

**Change this:**

            \# Step 2: Perform hybrid search  
            search\_results \= self.search\_engine.hybrid\_search(  
                query=rewritten\_query,  
                top\_k=10  \# Get top 10 results  
            )

**To this:**

            \# Step 2: Perform search based on strategy  
            search\_results \= \[\]  
            search\_top\_k \= 20 if self.use\_reranking else self.top\_k \# Get more docs if reranking  
              
            if self.retrieval\_strategy \== "hybrid":  
                search\_results \= self.search\_engine.hybrid\_search(  
                    query=rewritten\_query,  
                    top\_k=search\_top\_k  
                )  
            elif self.retrieval\_strategy \== "vector\_only":  
                search\_results \= self.search\_engine.vector\_only\_search(  
                    query=rewritten\_query,  
                    top\_k=search\_top\_k  
                )  
            else:  
                return f"Error: Unknown retrieval strategy '{self.retrieval\_strategy}'", 0, 0

            if not search\_results:  
                return "No relevant documents found for your question.", 0, 0

### **2\. Integrate Trustworthiness Metrics**

To run the TrustworthinessEvaluator per query, we need to add a method to it.

**File:** evaluation/trustworthiness\_evaluator.py

**Add this new method** inside the TrustworthinessEvaluator class:

    def evaluate\_single\_response(self, response: str, test\_case: Dict) \-\> Dict:  
        """Runs trustworthiness checks on a single query-response pair."""  
          
        \# 1\. Hallucination & Accuracy  
        hallucination\_analysis \= self.analyze\_hallucination(response, test\_case)  
          
        \# 2\. Response Quality (subset)  
        quality\_metrics \= {  
            'has\_specific\_info': self.check\_specific\_information(response),  
            'completeness\_score': self.assess\_completeness(response, test\_case)  
        }  
          
        return {  
            "hallucination\_score": hallucination\_analysis\['hallucination\_score'\],  
            "accuracy\_score": hallucination\_analysis\['accuracy\_score'\],  
            "completeness\_score": quality\_metrics\['completeness\_score'\],  
            "has\_specific\_info": quality\_metrics\['has\_specific\_info'\]  
        }

## **Stage 3: Implement Advanced Retrieval Logic (Exp 2 & 3\)**

### **1\. Add sentence-transformers dependency**

**File:** requirements.txt

**Add this line** to the file:

sentence-transformers

Then, **stop and run the installation** in your terminal:

pip install \-r requirements.txt

Continue to the next step *after* the installation is complete.

### **2\. Implement HyDE (Exp 2\)**

**File:** query\_processor.py

**Add this new method** inside the QueryProcessor class:

    def \_generate\_hypothetical\_answer(self, query: str) \-\> str:  
        """Generates a hypothetical answer to use as a search query."""  
        print("  Generating HyDE query...")  
        prompt \= f"""You are a helpful assistant. Generate a hypothetical document that provides a perfect answer to the following user query. Do not say it is hypothetical.

Query: {query}  
Answer:"""  
          
        try:  
            from openai import OpenAI  
            client \= OpenAI(api\_key=os.getenv("OPENAI\_API\_KEY"))  
              
            response \= client.chat.completions.create(  
                model="gpt-4o-mini", \# Use the mini model for this, it's fast  
                messages=\[{"role": "user", "content": prompt}\],  
                max\_tokens=300,  
                temperature=0.0  
            )  
            return response.choices\[0\].message.content.strip()  
        except Exception as e:  
            print(f"  HyDE generation failed: {e}. Falling back to original query.")  
            return query

**Modify process\_query** to use this method.

**Change this:**

            \# Step 1: Rewrite query for better retrieval  
            rewritten\_query \= self.\_rewrite\_query(query)  
            print(f"Rewritten query: {rewritten\_query}")

**To this:**

            \# Step 1: Rewrite query (or use HyDE)  
            retrieval\_query \= ""  
            if self.use\_hyde:  
                retrieval\_query \= self.\_generate\_hypothetical\_answer(query)  
            else:  
                retrieval\_query \= self.\_rewrite\_query(query)  
            print(f"Retrieval query: {retrieval\_query\[:150\]}...")

And **update the search\_results call** to use retrieval\_query:

            if self.retrieval\_strategy \== "hybrid":  
                search\_results \= self.search\_engine.hybrid\_search(  
                    query=retrieval\_query, \# Use retrieval\_query  
                    top\_k=search\_top\_k  
                )  
            elif self.retrieval\_strategy \== "vector\_only":  
                search\_results \= self.search\_engine.vector\_only\_search(  
                    query=retrieval\_query, \# Use retrieval\_query  
                    top\_k=search\_top\_k  
                )

### **3\. Implement Reranking (Exp 3\)**

**File:** query\_processor.py

**Add this new method** inside the QueryProcessor class:

    def \_rerank\_results(self, query: str, search\_results: List\[Dict\]) \-\> List\[Dict\]:  
        """Reranks search results using a CrossEncoder model."""  
        if not self.use\_reranking or not search\_results:  
            return search\_results\[:self.top\_k\] \# Return top\_k if not reranking  
              
        print(f"  Reranking {len(search\_results)} results...")  
        \# Lazy-load the reranker model  
        if self.reranker is None:  
            print("  Loading reranker model (one-time)...")  
            try:  
                self.reranker \= CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  
            except Exception as e:  
                print(f"  Failed to load reranker: {e}. Returning top {self.top\_k}.")  
                return search\_results\[:self.top\_k\]

        \# Create pairs for the model  
        pairs \= \[(query, result\['content'\]) for result in search\_results\]  
          
        try:  
            \# Get scores  
            scores \= self.reranker.predict(pairs)  
              
            \# Add scores to results and sort  
            for i in range(len(search\_results)):  
                search\_results\[i\]\['rerank\_score'\] \= scores\[i\]  
                  
            sorted\_results \= sorted(search\_results, key=lambda x: x\['rerank\_score'\], reverse=True)  
              
            print("  Reranking complete.")  
            return sorted\_results\[:self.top\_k\] \# Return new top\_k  
              
        except Exception as e:  
            print(f"  Reranking failed: {e}. Returning top {self.top\_k}.")  
            return search\_results\[:self.top\_k\]

**Modify process\_query** to call this method *after* search and *before* generation (this was done in Stage 1, Item 1).

## **Stage 4: Implement Embedding Model Experiment (Exp 1\)**

This requires updating the *batch setup* process to create a new, separate index.

### **1\. Modify document\_processor.py**

Pass the embedding\_model parameter through to the EmbeddingGenerator.

**File:** document\_processor.py

**Change this:**

class DocumentProcessor:  
    def \_\_init\_\_(self):  
        self.file\_handler \= FileHandler()  
        self.embedding\_generator \= EmbeddingGenerator()  
        self.index\_builder \= SearchIndexBuilder()  
        self.batch\_manager \= BatchManager()

**To this:**

class DocumentProcessor:  
    def \_\_init\_\_(self, embedding\_model: str \= "text-embedding-3-small"):  
        self.file\_handler \= FileHandler()  
        self.embedding\_generator \= EmbeddingGenerator(model\_name=embedding\_model) \# Pass model name  
        self.index\_builder \= SearchIndexBuilder(embedding\_generator=self.embedding\_generator) \# Pass generator  
        self.batch\_manager \= BatchManager()

### **2\. Update utils/search.py**

To allow the EmbeddingGenerator to be configurable, we must update SearchIndexBuilder to accept it as a parameter.

**File:** utils/search.py

**Change SearchIndexBuilder.\_\_init\_\_ from:**

    def \_\_init\_\_(self):  
        self.embedding\_generator \= None

**To:**

    def \_\_init\_\_(self, embedding\_generator=None):  
        if embedding\_generator:  
            self.embedding\_generator \= embedding\_generator  
        else:  
            from utils.embeddings import EmbeddingGenerator  
            self.embedding\_generator \= EmbeddingGenerator()

*(This allows DocumentProcessor to inject the correct generator)*

### **3\. Modify setup\_batch.py**

Add a CLI argument to specify the embedding model.

**File:** setup\_batch.py

**Add this line** in the main() function, after parser.add\_argument("--source", ...):

    parser.add\_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model to use (e.g., text-embedding-3-small, text-embedding-3-large)")

And **change this line** in main():

        document\_processor \= DocumentProcessor()

**To this:**

        document\_processor \= DocumentProcessor(embedding\_model=args.embedding\_model)

### **4\. ⚠️ CRITICAL USER ACTION: Create the New Batch**

To run Exp 1, you must create the new batch with the large embedding model. **Run this in your terminal before proceeding:**

python setup\_batch.py critical\_illness\_large \--source documents/critical\_illness \--embedding-model text-embedding-3-large

This will create a new batch folder batches/critical\_illness\_large with 3072-dimension vectors.

## **Stage 5: Create the Experimentation Harness Script**

All modifications are complete. Now, create the new script to run all experiments, incorporating all your requirements.

### **Experiment Details (for Copilot context)**

This is what we are building the harness script to run:

| Experiment | Pipeline Modification | Hypothesis | Key Metrics |
| :---- | :---- | :---- | :---- |
| **Control 1** | gpt-4o-mini (No Retrieval) | Establishes a 'no-RAG' floor. Answers will be general and prone to hallucination. | Faithfulness (→ 0.0) |
| **Control 2** | **Baseline:** text-embedding-3-small \+ Hybrid Search \+ gpt-4o-mini | This is the system to beat. Should be well-balanced. | All RAGAS Scores |
| **Exp 1** | **Embedding:** text-embedding-3-large | Larger model \= better semantic matching and context. | Context Precision (↑), Recall (↑) |
| **Exp 2** | **Query:** HyDE Transformation | Hypothetical answer is semantically closer to context chunks. | Context Precision (↑), Relevance (↑) |
| **Exp 3** | **Reranking:** Retrieve Top 20 → Rerank to Top 5 | Dedicated reranker is better at fine-tuning relevance. | Context Precision (↑), Faithfulness (↑) |
| **Exp 4** | **Generation:** gpt-5-preview (vs. mini) | More powerful generator \= better answer synthesis from same context. | Faithfulness (↑), Relevance (↑) |
| **Exp 5** | **Retrieval:** Vector-Only Search (FAISS) | Validates hybrid approach. Should perform *worse* than baseline. | Context Precision (↓), Recall (↓) |

### **The Harness Script**

**New File:** evaluation/run\_experiments.py

import json  
import time  
import os  
import sys  
from pathlib import Path  
from dataclasses import asdict  
from typing import List, Dict, Any

\# Add project root to path  
sys.path.insert(0, str(Path(\_\_file\_\_).parent.parent))

from batch\_manager import BatchManager  
from query\_processor import QueryProcessor  
from evaluation.industry\_standard\_evaluator import IndustryStandardEvaluator, StandardRAGMetrics  
from evaluation.trustworthiness\_evaluator import TrustworthinessEvaluator  
from evaluation.llm\_baseline\_comparator import BaselineLLMComparator

\# \--- Experiment Configuration \---  
\# This dictionary defines all experiments to run, based on the RAG Pipeline comparison doc.  
\# 'params' are passed directly to QueryProcessor.\_\_init\_\_  
EXPERIMENT\_CONFIGS \= {  
    "Control\_1\_LLM\_Baseline": {  
        "description": "gpt-4o-mini with no retrieval. Establishes a 'no-RAG' floor.",  
        "type": "baseline",  
        "processor\_class": BaselineLLMComparator,  
        "batch\_id": None,  
        "params": {}  
    },  
    "Control\_2\_RAG\_Baseline": {  
        "description": "Current system: text-embedding-3-small \+ Hybrid Search \+ gpt-4o-mini.",  
        "type": "rag",  
        "processor\_class": QueryProcessor,  
        "batch\_id": "critical\_illness",  
        "params": {  
            "generation\_model": "gpt-4o-mini",  
            "retrieval\_strategy": "hybrid",  
            "use\_hyde": False,  
            "use\_reranking": False  
        }  
    },  
    "Exp\_1\_Embedding\_Large": {  
        "description": "Tests text-embedding-3-large. Hypothesis: Improved semantic matching.",  
        "type": "rag",  
        "processor\_class": QueryProcessor,  
        "batch\_id": "critical\_illness\_large", \# Requires \--embedding-model text-embedding-3-large  
        "params": {  
            "generation\_model": "gpt-4o-mini",  
            "retrieval\_strategy": "hybrid"  
        }  
    },  
    "Exp\_2\_Query\_HyDE": {  
        "description": "Tests HyDE query transformation. Hypothesis: Bridges query-document gap.",  
        "type": "rag",  
        "processor\_class": QueryProcessor,  
        "batch\_id": "critical\_illness",  
        "params": {  
            "generation\_model": "gpt-4o-mini",  
            "retrieval\_strategy": "hybrid",  
            "use\_hyde": True  
        }  
    },  
    "Exp\_3\_Reranking": {  
        "description": "Tests Retrieve-then-Rerank (Top 5 from 20). Hypothesis: Improves context quality.",  
        "type": "rag",  
        "processor\_class": QueryProcessor,  
        "batch\_id": "critical\_illness",  
        "params": {  
            "generation\_model": "gpt-4o-mini",  
            "retrieval\_strategy": "hybrid",  
            "use\_reranking": True  
        }  
    },  
    "Exp\_4\_Gen\_Model\_GPT5": {  
        "description": "Tests gpt-5-preview as generator. Hypothesis: Improves answer synthesis.",  
        "type": "rag",  
        "processor\_class": QueryProcessor,  
        "batch\_id": "critical\_illness",  
        "params": {  
            "generation\_model": "gpt-5-preview", \# NOTE: Change this to a valid model  
            "retrieval\_strategy": "hybrid"  
        }  
    },  
    "Exp\_5\_Retrieval\_VectorOnly": {  
        "description": "Tests Vector-Only (FAISS) search. Hypothesis: Performs worse, proving value of hybrid.",  
        "type": "rag",  
        "processor\_class": QueryProcessor,  
        "batch\_id": "critical\_illness",  
        "params": {  
            "generation\_model": "gpt-4o-mini",  
            "retrieval\_strategy": "vector\_only" \# Use vector\_only  
        }  
    }  
}  
\# \------------------------------

def load\_queries() \-\> List\[Dict\[str, Any\]\]:  
    """Load test cases from ground\_truth.json and custom\_queries.json"""  
    queries \= \[\]  
      
    \# Load ground truth queries  
    try:  
        with open("evaluation/ground\_truth.json", 'r') as f:  
            data \= json.load(f)  
        ground\_queries \= data.get("test\_cases", \[\])  
        queries.extend(ground\_queries)  
        print(f"Loaded {len(ground\_queries)} queries from ground\_truth.json")  
    except FileNotFoundError:  
        print("evaluation/ground\_truth.json not found.")  
          
    \# Load custom queries  
    try:  
        with open("evaluation/custom\_queries.json", 'r') as f:  
            data \= json.load(f)  
        custom\_queries \= data.get("queries", \[\])  
        queries.extend(custom\_queries)  
        print(f"Loaded {len(custom\_queries)} queries from custom\_queries.json")  
    except FileNotFoundError:  
        print("evaluation/custom\_queries.json not found.")  
          
    if not queries:  
        print("No test cases found. Exiting.")  
        return None  
          
    print(f"Total queries to run: {len(queries)}")  
    return queries

def run\_experiment(exp\_name: str, config: dict, queries: list,   
                   ragas\_evaluator: IndustryStandardEvaluator,   
                   trust\_evaluator: TrustworthinessEvaluator,   
                   batch\_manager: BatchManager):  
    """Runs a single experiment configuration against all queries."""  
      
    print(f"\\n{'='\*30}\\n\[RUNNING\] {exp\_name}\\n{'='\*30}")  
    print(f"  Desc: {config\['description'\]}")  
      
    \# Initialize the correct processor for this experiment  
    processor \= None  
    if config\["type"\] \== "rag":  
        processor \= config\["processor\_class"\](batch\_manager, \*\*config\["params"\])  
    elif config\["type"\] \== "baseline":  
        processor \= config\["processor\_class"\]()

    if not processor:  
        print(f"  \[ERROR\] Could not initialize processor for {exp\_name}")  
        return None

    all\_metrics\_details \= \[\] \# Store individual results  
    total\_start\_time \= time.time()

    for i, test\_case in enumerate(queries):  
        query \= test\_case\['query'\]  
        print(f"  Query {i+1}/{len(queries)}: {query\[:50\]}...")  
          
        query\_result \= {  
            "query\_id": test\_case.get('id', f'query\_{i}'),  
            "query": query,  
            "answer": "",  
            "metrics": {},  
            "error": None  
        }  
          
        try:  
            rag\_response \= ""  
            retrieved\_contexts \= \[\]  
            generation\_tokens \= 0  
            latency \= 0  
              
            if config\["type"\] \== "rag":  
                rag\_response, generation\_tokens, latency \= processor.process\_query(query, config\["batch\_id"\])  
                  
                if "Error processing query" in rag\_response:  
                    raise Exception(rag\_response)  
                      
                \# Get contexts (must access search\_engine, which is loaded \*during\* process\_query)  
                if processor.search\_engine:  
                    search\_query \= query  
                    if config\["params"\].get("use\_hyde", False):  
                        search\_query \= processor.\_generate\_hypothetical\_answer(query)  
                      
                    search\_k \= 20 if config\["params"\].get("use\_reranking", False) else config\["params"\].get("top\_k", 10\)  
                    strategy \= config\["params"\].get("retrieval\_strategy", "hybrid")

                    chunks \= \[\]  
                    if strategy \== "hybrid":  
                        chunks \= processor.search\_engine.hybrid\_search(search\_query, top\_k=search\_k)  
                    elif strategy \== "vector\_only":  
                        chunks \= processor.search\_engine.vector\_only\_search(search\_query, top\_k=search\_k)  
                      
                    \# Rerank if needed  
                    top\_chunks \= processor.\_rerank\_results(query, chunks)  
                    retrieved\_contexts \= \[c\['content'\] for c in top\_chunks\]  
                  
            elif config\["type"\] \== "baseline":  
                baseline\_result \= processor.get\_baseline\_response(query)  
                rag\_response \= baseline\_result\['response'\]  
                generation\_tokens \= baseline\_result.get('tokens\_used', 0\)  
                latency \= baseline\_result\['time\_seconds'\]  
                retrieved\_contexts \= \[\] \# No contexts for baseline

            \# Evaluate the response  
            \# 1\. RAGAS Metrics  
            ragas\_metrics \= ragas\_evaluator.evaluate\_rag\_response(  
                query=query,  
                answer=rag\_response,  
                retrieved\_contexts=retrieved\_contexts  
            )  
              
            \# 2\. Trustworthiness Metrics  
            trust\_metrics \= trust\_evaluator.evaluate\_single\_response(rag\_response, test\_case)  
              
            \# 3\. Combine all metrics  
            query\_result\["answer"\] \= rag\_response  
            query\_result\["metrics"\] \= {  
                \*\*asdict(ragas\_metrics),  
                \*\*trust\_metrics,  
                "generation\_tokens": generation\_tokens,  
                "latency\_seconds": latency  
            }

        except Exception as e:  
            error\_msg \= f"Query failed: {e}"  
            print(f"    \[ERROR\] {error\_msg}")  
            query\_result\["error"\] \= error\_msg  
          
        all\_metrics\_details.append(query\_result) \# Log detail regardless of success

    total\_end\_time \= time.time()  
      
    \# \--- Aggregate results \---  
    valid\_results \= \[r for r in all\_metrics\_details if r\["error"\] is None\]  
    if not valid\_results:  
        print(f"  \[ERROR\] No valid metrics collected for {exp\_name}")  
        return None

    \# Calculate averages  
    def get\_avg(metric\_name):  
        return sum(r\["metrics"\]\[metric\_name\] for r in valid\_results) / len(valid\_results)

    summary \= {  
        "experiment\_name": exp\_name,  
        "description": config\["description"\],  
        "averages": {  
            "avg\_ragas\_score": get\_avg("ragas\_score"),  
            "avg\_faithfulness": get\_avg("faithfulness"),  
            "avg\_answer\_relevance": get\_avg("answer\_relevance"),  
            "avg\_context\_precision": get\_avg("context\_precision"),  
            "avg\_context\_recall": get\_avg("context\_recall"),  
            "avg\_hallucination\_score": get\_avg("hallucination\_score"),  
            "avg\_accuracy\_score": get\_avg("accuracy\_score"),  
            "avg\_latency\_seconds": get\_avg("latency\_seconds"),  
            "avg\_generation\_tokens": get\_avg("generation\_tokens"),  
        },  
        "total\_time\_s": total\_end\_time \- total\_start\_time,  
        "queries\_run": len(valid\_results),  
        "queries\_failed": len(queries) \- len(valid\_results),  
        "params": config\["params"\],  
        "individual\_query\_results": all\_metrics\_details \# Add the detailed log  
    }  
      
    print(f"  \[DONE\] Avg. RAGAS: {summary\['averages'\]\['avg\_ragas\_score'\]:.4f} | Avg. Latency: {summary\['averages'\]\['avg\_latency\_seconds'\]:.2f}s")  
    return summary

def main():  
    print("Starting RAG Pipeline Experiment Harness...")  
      
    \# \--- Check for large embedding batch \---  
    batch\_manager \= BatchManager()  
    if "Exp\_1\_Embedding\_Large" in EXPERIMENT\_CONFIGS:  
        if "critical\_illness\_large" not in batch\_manager.list\_batches():  
            print("="\*80)  
            print("WARNING: 'critical\_illness\_large' batch not found.")  
            print("Experiment 'Exp\_1\_Embedding\_Large' will be skipped.")  
            print("To run it, create the batch first:")  
            print("python setup\_batch.py critical\_illness\_large \--source documents/critical\_illness \--embedding-model text-embedding-3-large")  
            print("="\*80)  
            del EXPERIMENT\_CONFIGS\["Exp\_1\_Embedding\_Large"\]  
      
    queries \= load\_queries()  
    if not queries:  
        return

    \# Initialize BOTH evaluators  
    ragas\_evaluator \= IndustryStandardEvaluator(use\_llm\_judge=True)  
    trust\_evaluator \= TrustworthinessEvaluator(None, None) \# Doesn't need processor for single eval  
      
    all\_results \= \[\]  
    for exp\_name, config in EXPERIMENT\_CONFIGS.items():  
        result \= run\_experiment(exp\_name, config, queries, ragas\_evaluator, trust\_evaluator, batch\_manager)  
        if result:  
            all\_results.append(result)

    \# \--- Print Final Report \---  
    print("\\n\\n" \+ "="\*80)  
    print(" RAG PIPELINE EXPERIMENTATION \- FINAL RESULTS ")  
    print("="\*80)  
      
    \# Sort by RAGAS score  
    all\_results.sort(key=lambda x: x\['averages'\]\['avg\_ragas\_score'\], reverse=True)  
      
    print(f"{'Experiment':\<30} | {'RAGAS Score':\<12} | {'Faithfulness':\<13} | {'Hallucination':\<14} | {'Latency (s)':\<12} | {'Tokens':\<8}")  
    print("-" \* 95\)  
      
    for res in all\_results:  
        avg \= res\['averages'\]  
        print(f"{res\['experiment\_name'\]:\<30} | {avg\['avg\_ragas\_score'\]:\<12.4f} | {avg\['avg\_faithfulness'\]:\<13.4f} | {avg\['avg\_hallucination\_score'\]:\<14.4f} | {avg\['avg\_latency\_seconds'\]:\<12.2f} | {avg\['avg\_generation\_tokens'\]:\<8.0f}")

    \# Save results to JSON  
    timestamp \= time.strftime("%Y%m%d\_%H%M%S")  
    output\_file \= f"evaluation/results/experiment\_report\_{timestamp}.json"  
      
    \# Make sure parent directory exists  
    Path(output\_file).parent.mkdir(parents=True, exist\_ok=True)  
      
    with open(output\_file, 'w') as f:  
        json.dump(all\_results, f, indent=2)  
          
    print("="\*80)  
    print(f"Detailed results saved to: {output\_file}")

if \_\_name\_\_ \== "\_\_main\_\_":  
    main()

## **Stage 6: Future Work (Out of Scope)**

The following experiments from your plan are more complex and are not included in this implementation. They would require significant new components:

* **Exp 6 (Knowledge Graph):** Requires building a graph database (e.g., Neo4j) and a graph retrieval component.  
* **Exp 7 (Multi-Stage RAG):** Requires a more complex orchestration logic in QueryProcessor to run sequential queries and refinement.

This plan provides a clear, staged path to make your RAG pipeline fully configurable and creates a robust script (evaluation/run\_experiments.py) to automatically run and score all your defined experiments.