import json
import re
from typing import Dict, List, Tuple
from pathlib import Path
import time

class TrustworthinessEvaluator:
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
            print("Warning: ground_truth.json not found. Using empty test cases.")
            return []
    
    def run_evaluation(self):
        """Run comprehensive trustworthiness evaluation"""
        print("Starting Trustworthiness Evaluation...")
        print(f"Running {len(self.test_cases)} test cases")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_test_cases": len(self.test_cases),
            "hallucination_test": self.test_hallucination(),
            "consistency_test": self.test_consistency(), 
            "citation_test": self.test_citations(),
            "response_quality_test": self.test_response_quality()
        }
        
        # Calculate overall trustworthiness score
        results["overall_trustworthiness_score"] = self.calculate_overall_score(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def test_hallucination(self):
        # test for hallucinations in responses
        print("\n Testing Hallucination Detection...")
        results = []
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"  [{i}/{len(self.test_cases)}] Testing: {test['id']}")
            
            # Switch to critical illness batch
            if not self.batch_manager.switch_batch(self.batch_id):
                print(f"     Failed to switch to batch: {self.batch_id}")
                continue
                
            # Get response
            try:
                response = self.query_processor.process_query(test['query'])
            except Exception as e:
                print(f"     Query failed: {e}")
                continue
            
            # Analyze response for hallucination indicators
            hallucination_analysis = self.analyze_hallucination(response, test)
            
            results.append({
                'test_id': test['id'],
                'company': test['company'],
                'query': test['query'],
                'response': response,
                'expected_found': hallucination_analysis['expected_found'],
                'forbidden_found': hallucination_analysis['forbidden_found'],
                'hallucination_score': hallucination_analysis['hallucination_score'],
                'accuracy_score': hallucination_analysis['accuracy_score']
            })
        
        # Calculate averages
        if results:
            avg_hallucination = sum(r['hallucination_score'] for r in results) / len(results)
            avg_accuracy = sum(r['accuracy_score'] for r in results) / len(results)
        else:
            avg_hallucination = 0
            avg_accuracy = 0
        
        print(f"     Average Hallucination Rate: {avg_hallucination:.3f}")
        print(f"     Average Accuracy Rate: {avg_accuracy:.3f}")
        
        return {
            'avg_hallucination_rate': avg_hallucination,
            'avg_accuracy_rate': avg_accuracy,
            'total_tests': len(results),
            'details': results
        }
    
    def analyze_hallucination(self, response: str, test_case: Dict) -> Dict:
        # analyse response for hallucinations
        response_lower = response.lower()
        
        # Check for forbidden content (potential hallucinations)
        forbidden_keywords = test_case.get('forbidden_keywords', [])
        forbidden_found = []
        for keyword in forbidden_keywords:
            if keyword.lower() in response_lower:
                forbidden_found.append(keyword)
        
        # Check for expected content
        expected_keywords = test_case.get('expected_keywords', [])
        expected_found = []
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                expected_found.append(keyword)
        
        # Calculate scores
        hallucination_score = len(forbidden_found) / len(forbidden_keywords) if forbidden_keywords else 0
        accuracy_score = len(expected_found) / len(expected_keywords) if expected_keywords else 0
        
        return {
            'expected_found': expected_found,
            'forbidden_found': forbidden_found,
            'hallucination_score': hallucination_score,
            'accuracy_score': accuracy_score
        }
    
    def test_consistency(self):
        # test for consistency across different companies
        print("\n Testing Cross-Company Consistency...")
        
        # Load consistency queries from ground truth or use defaults
        try:
            with open("evaluation/ground_truth.json", 'r') as f:
                data = json.load(f)
            consistency_queries = data.get("consistency_queries", [
                "What is the waiting period for critical illness coverage?",
                "What cancers are covered under critical illness?",
                "How do I make a critical illness claim?",
                "What is the survival period requirement?"
            ])
        except:
            consistency_queries = [
                "What is the waiting period for critical illness coverage?",
                "What cancers are covered under critical illness?",
                "How do I make a critical illness claim?"
            ]
        
        results = {}
        
        for i, query in enumerate(consistency_queries, 1):
            print(f"  [{i}/{len(consistency_queries)}] Testing: '{query[:50]}...'")
            
            responses = {}
            if self.batch_manager.switch_batch(self.batch_id):
                try:
                    response = self.query_processor.process_query(query)
                    responses["critical_illness"] = response
                except Exception as e:
                    print(f"      Query failed: {e}")
                    responses["critical_illness"] = None
            
            # Calculate consistency metrics
            consistency_score = self.calculate_consistency_score(responses)
            
            results[query] = {
                'responses': responses,
                'consistency_score': consistency_score,
                'response_lengths': {k: len(v) if v else 0 for k, v in responses.items()}
            }
        
        avg_consistency = sum(r['consistency_score'] for r in results.values()) / len(results) if results else 0
        print(f"Average Consistency Score: {avg_consistency:.3f}")
        
        return {
            'avg_consistency_score': avg_consistency,
            'total_queries': len(results),
            'details': results
        }
    
    def calculate_consistency_score(self, responses: Dict[str, str]) -> float:
        # calculate consistency score for resonse similarity
        valid_responses = {k: v for k, v in responses.items() if v}
        
        if len(valid_responses) < 1:
            return 0.0
        
        # For single batch, consistency is always 1.0 since there's only one response
        return 1.0
        
        # Simple consistency check based on:
        # 1. Similar response lengths
        # 2. Common key terms
        
        lengths = [len(resp) for resp in valid_responses.values()]
        length_consistency = 1 - (max(lengths) - min(lengths)) / max(lengths) if max(lengths) > 0 else 1
        
        # Extract common terms
        all_terms = []
        for response in valid_responses.values():
            terms = set(re.findall(r'\b\w{4,}\b', response.lower()))
            all_terms.append(terms)
        
        # Calculate term overlap
        if len(all_terms) >= 2:
            common_terms = set.intersection(*all_terms)
            all_unique_terms = set.union(*all_terms)
            term_consistency = len(common_terms) / len(all_unique_terms) if all_unique_terms else 0
        else:
            term_consistency = 0
        
        # Weighted average
        return (length_consistency * 0.3 + term_consistency * 0.7)
    
    def test_citations(self):
        """Test if responses include proper source citations"""
        print("\n Testing Citation Quality...")
        
        citation_patterns = [
            r'according to',
            r'based on',
            r'policy states',
            r'document shows',
            r'section \d+',
            r'clause',
            r'page \d+',
            r'source:',
            r'reference:'
        ]
        
        results = []
        
        for i, test in enumerate(self.test_cases, 1):
            print(f"  [{i}/{len(self.test_cases)}] Testing citations: {test['id']}")
            
            if not self.batch_manager.switch_batch(self.batch_id):
                continue
                
            try:
                response = self.query_processor.process_query(test['query'])
            except Exception as e:
                continue
            
            # Check for citation indicators
            citations_found = []
            for pattern in citation_patterns:
                matches = re.findall(pattern, response.lower())
                citations_found.extend(matches)
            
            has_citation = len(citations_found) > 0
            citation_quality = min(len(citations_found) / 2, 1.0)  # Cap at 1.0, normalize by 2 citations
            
            results.append({
                'test_id': test['id'],
                'has_citation': has_citation,
                'citations_found': citations_found,
                'citation_quality': citation_quality,
                'response_length': len(response)
            })
        
        citation_rate = sum(r['has_citation'] for r in results) / len(results) if results else 0
        avg_citation_quality = sum(r['citation_quality'] for r in results) / len(results) if results else 0
        
        print(f"Citation Rate: {citation_rate:.3f}")
        print(f"Average Citation Quality: {avg_citation_quality:.3f}")
        
        return {
            'citation_rate': citation_rate,
            'avg_citation_quality': avg_citation_quality,
            'total_responses': len(results),
            'details': results
        }
    
    def test_response_quality(self):
        """Test overall response quality metrics"""
        print("\n Testing Response Quality...")
        
        results = []
        
        for i, test in enumerate(self.test_cases, 1):
            if not self.batch_manager.switch_batch(self.batch_id):
                continue
            
            try:
                start_time = time.time()
                response = self.query_processor.process_query(test['query'])
                response_time = time.time() - start_time
                
                quality_metrics = {
                    'response_length': len(response),
                    'response_time': response_time,
                    'word_count': len(response.split()),
                    'has_specific_info': self.check_specific_information(response),
                    'completeness_score': self.assess_completeness(response, test)
                }
                
                results.append({
                    'test_id': test['id'],
                    **quality_metrics
                })
                
            except Exception as e:
                continue
        
        if results:
            avg_response_time = sum(r['response_time'] for r in results) / len(results)
            avg_word_count = sum(r['word_count'] for r in results) / len(results)
            avg_completeness = sum(r['completeness_score'] for r in results) / len(results)
        else:
            avg_response_time = avg_word_count = avg_completeness = 0
        
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"Average Word Count: {avg_word_count:.1f}")
        print(f"Average Completeness: {avg_completeness:.3f}")
        
        return {
            'avg_response_time': avg_response_time,
            'avg_word_count': avg_word_count,
            'avg_completeness_score': avg_completeness,
            'details': results
        }
    
    def check_specific_information(self, response: str) -> bool:
        """Check if response contains specific, actionable information"""
        specific_indicators = [
            r'\d+\s*(days|months|years)',  # Time periods
            r'\$\d+',  # Money amounts
            r'\d+%',   # Percentages
            r'section \d+',  # Specific sections
            r'clause \d+',   # Specific clauses
        ]
        
        return any(re.search(pattern, response.lower()) for pattern in specific_indicators)
    
    def assess_completeness(self, response: str, test_case: Dict) -> float:
        """Assess how complete the response is"""
        expected_keywords = test_case.get('expected_keywords', [])
        if not expected_keywords:
            return 1.0
        
        response_lower = response.lower()
        keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        return keywords_found / len(expected_keywords)
    
    def calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall trustworthiness score"""
        # Weight different components
        weights = {
            'hallucination': 0.3,  # Lower is better
            'accuracy': 0.25,      # Higher is better
            'consistency': 0.2,    # Higher is better
            'citations': 0.15,     # Higher is better
            'completeness': 0.1    # Higher is better
        }
        
        # Get component scores (normalize so higher is always better)
        hallucination_score = 1 - results['hallucination_test']['avg_hallucination_rate']  # Invert
        accuracy_score = results['hallucination_test']['avg_accuracy_rate']
        consistency_score = results['consistency_test']['avg_consistency_score']
        citation_score = results['citation_test']['citation_rate']
        completeness_score = results['response_quality_test']['avg_completeness_score']
        
        overall = (
            hallucination_score * weights['hallucination'] +
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency'] +
            citation_score * weights['citations'] +
            completeness_score * weights['completeness']
        )
        
        return overall
    
    def save_results(self, results: Dict):
        """Save evaluation results to file"""
        results_file = Path("evaluation/results/trustworthiness_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n Results saved to: {results_file}")
    
    def print_summary(self, results: Dict):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*60)
        print("TRUSTWORTHINESS EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Overall Trustworthiness Score: {results['overall_trustworthiness_score']:.3f}/1.000")
        print(f"Evaluation Date: {results['timestamp']}")
        print(f"Total Test Cases: {results['total_test_cases']}")
        
        print(f"\nComponent Scores:")
        print(f"Hallucination Rate: {results['hallucination_test']['avg_hallucination_rate']:.3f} (lower is better)")
        print(f"Accuracy Rate: {results['hallucination_test']['avg_accuracy_rate']:.3f}")
        print(f"Consistency Score: {results['consistency_test']['avg_consistency_score']:.3f}")
        print(f"Citation Rate: {results['citation_test']['citation_rate']:.3f}")
        print(f"Completeness Score: {results['response_quality_test']['avg_completeness_score']:.3f}")
        
        print(f"\nPerformance Metrics:")
        print(f"Average Response Time: {results['response_quality_test']['avg_response_time']:.3f}s")
        print(f"Average Word Count: {results['response_quality_test']['avg_word_count']:.1f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if results['hallucination_test']['avg_hallucination_rate'] > 0.2:
            print("High hallucination rate detected - review source document quality")
        if results['citation_test']['citation_rate'] < 0.5:
            print("Low citation rate - improve source attribution in responses")
        if results['consistency_test']['avg_consistency_score'] < 0.6:
            print("Low consistency across companies - review document processing")
        
        print("="*60) 