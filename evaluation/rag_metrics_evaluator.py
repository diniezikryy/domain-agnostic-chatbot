import json
import time
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np

class RAGMetricsEvaluator:
    def __init__(self, query_processor, batch_manager):
        self.query_processor = query_processor
        self.batch_manager = batch_manager
        self.test_cases = self.load_test_cases()
        self.batch_id = "critical_illness"
    
    def load_test_cases(self):
        """Load ground truth test cases"""
        try:
            with open("evaluation/ground_truth.json", 'r') as f:
                data = json.load(f)
            return data.get("test_cases", [])
        except FileNotFoundError:
            return []
    
    def run_full_evaluation(self):
        """Run comprehensive RAG metrics evaluation"""
        print("Starting RAG Pipeline Metrics Evaluation...")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(self.test_cases),
            "retrieval_metrics": self.evaluate_retrieval_metrics(),
            "generation_metrics": self.evaluate_generation_metrics(),
            "overall_metrics": {}
        }
        
        # Calculate overall scores
        results["overall_metrics"] = self.calculate_overall_metrics(results)
        
        # Save results
        self.save_results(results)
        return results
    
    def evaluate_retrieval_metrics(self):
        """Evaluate retrieval-specific metrics: Hit Rate, MRR, Precision@K, Recall@K"""
        print("\n=== RETRIEVAL METRICS ===")
        
        hit_rates = []
        mrr_scores = []
        precision_at_k = []
        recall_at_k = []
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] Evaluating retrieval: {test['id']}")
            
            if not self.batch_manager.switch_batch(self.batch_id):
                continue
            
            # Get search results directly from search engine
            try:
                search_engine = self.query_processor.search_engine
                if not search_engine:
                    self.query_processor._ensure_batch_loaded(self.batch_id)
                    search_engine = self.query_processor.search_engine
                
                search_results = search_engine.hybrid_search(test['query'], top_k=10)
                
                # Evaluate against expected keywords
                relevant_docs = self.identify_relevant_docs(search_results, test)
                
                # Hit Rate: Did we retrieve at least one relevant document?
                hit_rate = 1.0 if any(relevant_docs) else 0.0
                hit_rates.append(hit_rate)
                
                # MRR: Mean Reciprocal Rank
                mrr = self.calculate_mrr(relevant_docs)
                mrr_scores.append(mrr)
                
                # Precision@K and Recall@K
                precision_k = self.calculate_precision_at_k(relevant_docs, k=5)
                recall_k = self.calculate_recall_at_k(relevant_docs, test, k=5)
                
                precision_at_k.append(precision_k)
                recall_at_k.append(recall_k)
                
            except Exception as e:
                print(f"Error in retrieval evaluation: {e}")
                continue
        
        return {
            "hit_rate": np.mean(hit_rates) if hit_rates else 0,
            "mean_reciprocal_rank": np.mean(mrr_scores) if mrr_scores else 0,
            "precision_at_5": np.mean(precision_at_k) if precision_at_k else 0,
            "recall_at_5": np.mean(recall_at_k) if recall_at_k else 0,
            "total_queries": len(hit_rates)
        }
    
    def evaluate_generation_metrics(self):
        """Evaluate generation-specific metrics: Relevancy, Faithfulness"""
        print("\n=== GENERATION METRICS ===")
        
        relevancy_scores = []
        faithfulness_scores = []
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] Evaluating generation: {test['id']}")
            
            if not self.batch_manager.switch_batch(self.batch_id):
                continue
            
            try:
                response = self.query_processor.process_query(test['query'])
                
                # Relevancy: How well does response match expected content
                relevancy = self.calculate_relevancy(response, test)
                relevancy_scores.append(relevancy)
                
                # Faithfulness: How well response stays true to source documents
                faithfulness = self.calculate_faithfulness(response, test)
                faithfulness_scores.append(faithfulness)
                
            except Exception as e:
                print(f"Error in generation evaluation: {e}")
                continue
        
        return {
            "relevancy_score": np.mean(relevancy_scores) if relevancy_scores else 0,
            "faithfulness_score": np.mean(faithfulness_scores) if faithfulness_scores else 0,
            "total_responses": len(relevancy_scores)
        }
    
    def identify_relevant_docs(self, search_results: List[Dict], test_case: Dict) -> List[bool]:
        """Identify which retrieved documents are relevant"""
        relevant_docs = []
        expected_keywords = test_case.get('expected_keywords', [])
        
        for result in search_results:
            content = result.get('content', '').lower()
            # Document is relevant if it contains expected keywords
            is_relevant = any(keyword.lower() in content for keyword in expected_keywords)
            relevant_docs.append(is_relevant)
        
        return relevant_docs
    
    def calculate_mrr(self, relevant_docs: List[bool]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, is_relevant in enumerate(relevant_docs):
            if is_relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_precision_at_k(self, relevant_docs: List[bool], k: int = 5) -> float:
        """Calculate Precision@K"""
        if not relevant_docs or k == 0:
            return 0.0
        
        top_k = relevant_docs[:k]
        return sum(top_k) / len(top_k)
    
    def calculate_recall_at_k(self, relevant_docs: List[bool], test_case: Dict, k: int = 5) -> float:
        """Calculate Recall@K (simplified - assumes all expected keywords represent total relevant docs)"""
        if not relevant_docs or k == 0:
            return 0.0
        
        # Simplified: assume number of expected keywords represents total relevant documents
        total_relevant = len(test_case.get('expected_keywords', []))
        if total_relevant == 0:
            return 0.0
        
        retrieved_relevant = sum(relevant_docs[:k])
        return min(retrieved_relevant / total_relevant, 1.0)
    
    def calculate_relevancy(self, response: str, test_case: Dict) -> float:
        """Calculate how relevant the response is to the query"""
        expected_keywords = test_case.get('expected_keywords', [])
        if not expected_keywords:
            return 0.0
        
        response_lower = response.lower()
        keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        return keywords_found / len(expected_keywords)
    
    def calculate_faithfulness(self, response: str, test_case: Dict) -> float:
        """Calculate faithfulness (no hallucinations)"""
        forbidden_keywords = test_case.get('forbidden_keywords', [])
        if not forbidden_keywords:
            return 1.0
        
        response_lower = response.lower()
        hallucinations = sum(1 for keyword in forbidden_keywords if keyword.lower() in response_lower)
        
        # Return 1 - hallucination_rate
        return max(0.0, 1.0 - (hallucinations / len(forbidden_keywords)))
    
    def calculate_overall_metrics(self, results: Dict) -> Dict:
        """Calculate overall pipeline metrics"""
        retrieval = results["retrieval_metrics"]
        generation = results["generation_metrics"]
        
        # Weighted overall score
        overall_score = (
            retrieval["hit_rate"] * 0.2 +
            retrieval["mean_reciprocal_rank"] * 0.2 +
            retrieval["precision_at_5"] * 0.15 +
            retrieval["recall_at_5"] * 0.15 +
            generation["relevancy_score"] * 0.15 +
            generation["faithfulness_score"] * 0.15
        )
        
        return {
            "overall_rag_score": overall_score,
            "retrieval_score": (retrieval["hit_rate"] + retrieval["mean_reciprocal_rank"] + 
                              retrieval["precision_at_5"] + retrieval["recall_at_5"]) / 4,
            "generation_score": (generation["relevancy_score"] + generation["faithfulness_score"]) / 2
        }
    
    def save_results(self, results: Dict):
        """Save evaluation results"""
        results_file = Path("evaluation/results/rag_metrics_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def print_summary(self, results: Dict):
        """Print comprehensive metrics summary"""
        print("\n" + "="*60)
        print("RAG PIPELINE METRICS SUMMARY")
        print("="*60)
        
        print(f"Overall RAG Score: {results['overall_metrics']['overall_rag_score']:.3f}/1.000")
        print(f"Evaluation Date: {results['timestamp']}")
        print(f"Total Test Cases: {results['total_test_cases']}")
        
        print(f"\nRETRIEVAL METRICS:")
        r = results['retrieval_metrics']
        print(f"Hit Rate: {r['hit_rate']:.3f}")
        print(f"Mean Reciprocal Rank: {r['mean_reciprocal_rank']:.3f}")
        print(f"Precision@5: {r['precision_at_5']:.3f}")
        print(f"Recall@5: {r['recall_at_5']:.3f}")
        
        print(f"\nGENERATION METRICS:")
        g = results['generation_metrics']
        print(f"Relevancy Score: {g['relevancy_score']:.3f}")
        print(f"Faithfulness Score: {g['faithfulness_score']:.3f}")
        
        print(f"\nOVERALL SCORES:")
        o = results['overall_metrics']
        print(f"Retrieval Score: {o['retrieval_score']:.3f}")
        print(f"Generation Score: {o['generation_score']:.3f}")
        
        print("="*60)