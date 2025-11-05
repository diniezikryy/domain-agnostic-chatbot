#!/usr/bin/env python3
"""
RAG Experimentation Harness
Runs configurable experiments and evaluates RAG pipeline performance.
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_manager import BatchManager
from query_processor import QueryProcessor


class ExperimentRunner:
    """Manages and executes RAG experiments."""
    
    def __init__(self, batch_id: str, output_dir: str = "evaluation/results"):
        self.batch_id = batch_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_manager = BatchManager()
        
        # Verify batch exists
        if not self.batch_manager.get_batch_info(batch_id):
            raise ValueError(f"Batch '{batch_id}' not found")
    
    def load_experiments(self, experiments_file: str = "evaluation/experiments.json") -> List[Dict]:
        """Load experiment configurations."""
        with open(experiments_file, 'r') as f:
            data = json.load(f)
        return data.get('experiments', [])
    
    def load_test_queries(self, queries_file: str = "evaluation/test_queries.json") -> List[Dict]:
        """Load test queries with ground truth."""
        with open(queries_file, 'r') as f:
            data = json.load(f)
        return data.get('queries', [])
    
    def run_experiment(self, experiment: Dict, queries: List[Dict]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        exp_name = experiment['name']
        exp_config = experiment['config']
        
        print(f"\n{'='*80}")
        print(f"Running Experiment: {exp_name}")
        print(f"Description: {experiment['description']}")
        print(f"Config: {json.dumps(exp_config, indent=2)}")
        print(f"{'='*80}\n")
        
        # Initialize query processor with experiment config
        query_processor = QueryProcessor(self.batch_manager, config=exp_config)
        
        results = {
            'name': exp_name,
            'description': experiment['description'],
            'config': exp_config,
            'query_results': [],
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Process each query
        for i, query_data in enumerate(queries, 1):
            query = query_data['query']
            print(f"\n[{i}/{len(queries)}] Processing: {query}")
            
            start_time = time.time()
            try:
                response = query_processor.process_query(query, self.batch_id)
                response_time = time.time() - start_time
                
                query_result = {
                    'query': query,
                    'response': response,
                    'response_time': response_time,
                    'ground_truth': query_data.get('ground_truth', ''),
                    'expected_behavior': query_data.get('expected_behavior', ''),
                    'success': True
                }
                
                print(f"✓ Completed in {response_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                query_result = {
                    'query': query,
                    'response': None,
                    'response_time': None,
                    'error': str(e),
                    'success': False
                }
            
            results['query_results'].append(query_result)
        
        # Calculate aggregate metrics
        results['metrics'] = self._calculate_metrics(results['query_results'])
        
        return results
    
    def _calculate_metrics(self, query_results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for an experiment."""
        metrics = {}
        
        # Basic metrics
        successful_queries = [r for r in query_results if r.get('success', False)]
        metrics['success_rate'] = len(successful_queries) / len(query_results) if query_results else 0.0
        
        response_times = [r['response_time'] for r in successful_queries if r.get('response_time')]
        metrics['avg_response_time'] = sum(response_times) / len(response_times) if response_times else 0.0
        metrics['min_response_time'] = min(response_times) if response_times else 0.0
        metrics['max_response_time'] = max(response_times) if response_times else 0.0
        
        # Simple quality metrics (can be enhanced with RAGAS)
        metrics['total_queries'] = len(query_results)
        metrics['successful_queries'] = len(successful_queries)
        metrics['failed_queries'] = len(query_results) - len(successful_queries)
        
        # Check for citations in responses
        responses_with_citations = sum(
            1 for r in successful_queries 
            if r.get('response') and '[Source' in r['response']
        )
        metrics['citation_rate'] = responses_with_citations / len(successful_queries) if successful_queries else 0.0
        
        return metrics
    
    def run_all_experiments(
        self, 
        experiments_file: str = "evaluation/experiments.json",
        queries_file: str = "evaluation/test_queries.json",
        selected_experiment: Optional[str] = None
    ) -> List[Dict]:
        """Run all experiments or a specific one."""
        experiments = self.load_experiments(experiments_file)
        queries = self.load_test_queries(queries_file)
        
        # Filter to selected experiment if specified
        if selected_experiment:
            experiments = [e for e in experiments if e['name'] == selected_experiment]
            if not experiments:
                raise ValueError(f"Experiment '{selected_experiment}' not found")
        
        all_results = []
        
        for experiment in experiments:
            result = self.run_experiment(experiment, queries)
            all_results.append(result)
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = "experiment_results.json"):
        """Save experiment results to JSON file."""
        output_file = self.output_dir / filename
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'batch': self.batch_id,
            'num_experiments': len(results),
            'experiments': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        return output_file
    
    def print_leaderboard(self, results: List[Dict]):
        """Print formatted leaderboard of experiment results."""
        print("\n" + "="*80)
        print("RAG EXPERIMENT LEADERBOARD")
        print("="*80)
        print(f"Batch: {self.batch_id} | Experiments: {len(results)} | Date: {datetime.now().strftime('%Y-%m-%d')}")
        print()
        
        # Sort by success rate, then by avg response time
        sorted_results = sorted(
            results, 
            key=lambda x: (-x['metrics'].get('success_rate', 0), x['metrics'].get('avg_response_time', float('inf')))
        )
        
        # Print header
        header = f"{'Rank':<6} | {'Experiment':<20} | {'Success':<8} | {'Cite%':<7} | {'Avg Time':<9}"
        print(header)
        print("-"*80)
        
        # Print results
        for rank, result in enumerate(sorted_results, 1):
            metrics = result['metrics']
            name = result['name'][:20]
            success = f"{metrics.get('success_rate', 0)*100:.1f}%"
            citation = f"{metrics.get('citation_rate', 0)*100:.1f}%"
            avg_time = f"{metrics.get('avg_response_time', 0):.2f}s"
            
            row = f"{rank:<6} | {name:<20} | {success:<8} | {citation:<7} | {avg_time:<9}"
            print(row)
        
        print("="*80)
        print()
        
        # Print best performer details
        if sorted_results:
            best = sorted_results[0]
            print("🏆 Best Performer:")
            print(f"   Name: {best['name']}")
            print(f"   Description: {best['description']}")
            print(f"   Success Rate: {best['metrics'].get('success_rate', 0)*100:.1f}%")
            print(f"   Citation Rate: {best['metrics'].get('citation_rate', 0)*100:.1f}%")
            print(f"   Avg Response Time: {best['metrics'].get('avg_response_time', 0):.2f}s")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG experiments and evaluate performance"
    )
    
    parser.add_argument(
        "--batch",
        required=True,
        help="Batch ID to run experiments on"
    )
    
    parser.add_argument(
        "--experiment",
        help="Run specific experiment only (by name)"
    )
    
    parser.add_argument(
        "--queries",
        default="evaluation/test_queries.json",
        help="Path to test queries JSON file"
    )
    
    parser.add_argument(
        "--experiments-config",
        default="evaluation/experiments.json",
        help="Path to experiments configuration JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="evaluation/results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = ExperimentRunner(args.batch, args.output_dir)
        
        # Run experiments
        print(f"\nStarting RAG Experiments on batch: {args.batch}")
        print(f"Query file: {args.queries}")
        print(f"Experiments config: {args.experiments_config}")
        
        results = runner.run_all_experiments(
            experiments_file=args.experiments_config,
            queries_file=args.queries,
            selected_experiment=args.experiment
        )
        
        # Save results
        output_file = runner.save_results(results)
        
        # Print leaderboard
        runner.print_leaderboard(results)
        
        print(f"\n✓ Evaluation complete! Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
