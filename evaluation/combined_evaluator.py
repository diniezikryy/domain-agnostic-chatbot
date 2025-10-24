import json
import time
from typing import Dict, Any
from pathlib import Path
from .rag_metrics_evaluator import RAGMetricsEvaluator
from .trustworthiness_evaluator import TrustworthinessEvaluator

class CombinedEvaluator:
    """Combined evaluator for all RAG pipeline metrics"""
    
    def __init__(self, query_processor, batch_manager):
        self.rag_evaluator = RAGMetricsEvaluator(query_processor, batch_manager)
        self.trust_evaluator = TrustworthinessEvaluator(query_processor, batch_manager)
    
    def run_complete_evaluation(self):
        """Run all evaluations and combine results"""
        print("Starting Complete RAG Pipeline Evaluation...")
        
        # Run RAG metrics
        rag_results = self.rag_evaluator.run_full_evaluation()
        
        # Run trustworthiness metrics  
        trust_results = self.trust_evaluator.run_evaluation()
        
        # Combine results
        combined_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rag_metrics": rag_results,
            "trustworthiness_metrics": trust_results,
            "final_scores": self.calculate_final_scores(rag_results, trust_results)
        }
        
        # Save combined results
        self.save_combined_results(combined_results)
        
        # Print comprehensive summary
        self.print_complete_summary(combined_results)
        
        return combined_results
    
    def calculate_final_scores(self, rag_results: Dict, trust_results: Dict) -> Dict:
        """Calculate final combined scores"""
        
        # Extract key scores
        rag_score = rag_results["overall_metrics"]["overall_rag_score"]
        trust_score = trust_results["overall_trustworthiness_score"]
        
        # Calculate weighted final score
        final_score = (rag_score * 0.6) + (trust_score * 0.4)
        
        return {
            "final_pipeline_score": final_score,
            "rag_component_score": rag_score,
            "trustworthiness_component_score": trust_score,
            "grade": self.get_grade(final_score)
        }
    
    def get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9: return "A+"
        elif score >= 0.85: return "A"
        elif score >= 0.8: return "A-"
        elif score >= 0.75: return "B+"
        elif score >= 0.7: return "B"
        elif score >= 0.65: return "B-"
        elif score >= 0.6: return "C+"
        elif score >= 0.55: return "C"
        else: return "F"
    
    def save_combined_results(self, results: Dict):
        """Save combined evaluation results"""
        results_file = Path("evaluation/results/complete_evaluation.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nComplete results saved to: {results_file}")
    
    def print_complete_summary(self, results: Dict):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print("COMPLETE RAG PIPELINE EVALUATION SUMMARY")
        print("="*80)
        
        final = results["final_scores"]
        print(f"FINAL PIPELINE SCORE: {final['final_pipeline_score']:.3f}/1.000 (Grade: {final['grade']})")
        print(f"Evaluation Date: {results['timestamp']}")
        
        print(f"\nCOMPONENT BREAKDOWN:")
        print(f"RAG Metrics Score: {final['rag_component_score']:.3f}/1.000")
        print(f"Trustworthiness Score: {final['trustworthiness_component_score']:.3f}/1.000")
        
        # RAG Metrics Detail
        rag = results["rag_metrics"]
        print(f"\nRAG METRICS DETAIL:")
        print(f"Hit Rate: {rag['retrieval_metrics']['hit_rate']:.3f}")
        print(f"Mean Reciprocal Rank: {rag['retrieval_metrics']['mean_reciprocal_rank']:.3f}")
        print(f"Precision@5: {rag['retrieval_metrics']['precision_at_5']:.3f}")
        print(f"Recall@5: {rag['retrieval_metrics']['recall_at_5']:.3f}")
        print(f"Relevancy: {rag['generation_metrics']['relevancy_score']:.3f}")
        print(f"Faithfulness: {rag['generation_metrics']['faithfulness_score']:.3f}")
        
        # Trustworthiness Detail
        trust = results["trustworthiness_metrics"]
        print(f"\nTRUSTWORTHINESS DETAIL:")
        print(f"Hallucination Rate: {trust['hallucination_test']['avg_hallucination_rate']:.3f} (lower is better)")
        print(f"Accuracy Rate: {trust['hallucination_test']['avg_accuracy_rate']:.3f}")
        print(f"Consistency Score: {trust['consistency_test']['avg_consistency_score']:.3f}")
        print(f"Citation Rate: {trust['citation_test']['citation_rate']:.3f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if final['final_pipeline_score'] < 0.7:
            print("Pipeline needs significant improvement")
        if rag['retrieval_metrics']['hit_rate'] < 0.8:
            print("Improve document indexing and retrieval")
        if trust['hallucination_test']['avg_hallucination_rate'] > 0.2:
            print("Reduce hallucinations in responses")
        if trust['citation_test']['citation_rate'] < 0.5:
            print("Improve source attribution")
        
        print("="*80)