"""
Evaluation Test Suite for Insurance Policy Queries
Tests the RAG pipeline for correctness, balance, and evidence requirements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_manager import BatchManager
from query_processor import QueryProcessor
from typing import Dict, List, Any
import json
from datetime import datetime

class QueryEvaluator:
    """Evaluates RAG responses against expected criteria."""

    def __init__(self, batch_manager: BatchManager, query_processor: QueryProcessor):
        self.batch_manager = batch_manager
        self.query_processor = query_processor

    def evaluate_response(self, query: str, response: str, expected: Dict) -> Dict[str, Any]:
        """Evaluate a single response against expected criteria."""
        results = {
            "query": query,
            "response": response,
            "passed": True,
            "failures": [],
            "warnings": [],
            "score": 0,
            "max_score": 0
        }

        # Check 1: Must contain required terms
        if "must_contain" in expected:
            results["max_score"] += len(expected["must_contain"])
            for term in expected["must_contain"]:
                if term.lower() in response.lower():
                    results["score"] += 1
                else:
                    results["passed"] = False
                    results["failures"].append(f"Missing required term: '{term}'")

        # Check 2: Must cite both policies in comparisons
        if expected.get("must_cite_both_policies"):
            results["max_score"] += 2
            has_singlife = "singlife" in response.lower() or "sing life" in response.lower()
            has_fwd = "fwd" in response.lower()

            if has_singlife:
                results["score"] += 1
            else:
                results["failures"].append("Missing SingLife policy reference")
                results["passed"] = False

            if has_fwd:
                results["score"] += 1
            else:
                results["failures"].append("Missing FWD policy reference")
                results["passed"] = False

        # Check 3: Should acknowledge limitations
        if expected.get("should_acknowledge_limitations"):
            results["max_score"] += 1
            limitation_phrases = [
                "don't have", "doesn't provide", "cannot", "not available",
                "don't mention", "doesn't discuss", "not found", "insufficient",
                "do not provide", "documents do not"
            ]
            if any(phrase in response.lower() for phrase in limitation_phrases):
                results["score"] += 1
            else:
                results["warnings"].append("Should acknowledge data limitations when appropriate")

        # Check 4: Must have source citations (UPDATED - allows "no data" responses without citations)
        if expected.get("must_cite_sources"):
            results["max_score"] += 1

            # Check if this is a "no data found" response
            no_data_phrases = [
                "don't provide", "do not provide", "documents do not",
                "no information", "cannot provide", "not available"
            ]
            is_no_data_response = any(phrase in response.lower() for phrase in no_data_phrases)

            has_citation = "[source" in response.lower() or "source " in response.lower()

            # Pass if either has citations OR is clearly a no-data response
            if has_citation or is_no_data_response:
                results["score"] += 1
            else:
                results["failures"].append("Missing source citations")
                results["passed"] = False

        # Check 5: Should not make specific claims
        if "should_not_claim" in expected:
            results["max_score"] += len(expected["should_not_claim"])
            for claim in expected["should_not_claim"]:
                if claim.lower() not in response.lower():
                    results["score"] += 1
                else:
                    results["failures"].append(f"Made prohibited claim: '{claim}'")
                    results["passed"] = False

        # Check 6: Must acknowledge if comparing different profiles
        if expected.get("must_acknowledge_different_profiles"):
            results["max_score"] += 1
            profile_phrases = [
                "different", "not comparable", "different age", "different health",
                "different profile", "cannot directly compare"
            ]
            if any(phrase in response.lower() for phrase in profile_phrases):
                results["score"] += 1
            else:
                results["warnings"].append("Should note when comparing different customer profiles")

        # Check 7: Expected outcome matches
        if "expected_outcome" in expected:
            results["max_score"] += 1
            outcome = expected["expected_outcome"].lower()
            if outcome in response.lower():
                results["score"] += 1
            else:
                results["warnings"].append(f"Expected to mention: '{expected['expected_outcome']}'")

        # Calculate pass percentage
        if results["max_score"] > 0:
            results["pass_percentage"] = (results["score"] / results["max_score"]) * 100
        else:
            results["pass_percentage"] = 0

        return results


# Test Cases
TEST_CASES = [
    {
        "name": "Comparison - Unique Exclusions",
        "query": "What exclusions are present in SingLife's critical illness policy that are not in FWD's critical illness policy?",
        "expected": {
            "must_contain": ["singlife", "fwd", "exclusion"],
            "must_cite_both_policies": True,
            "must_cite_sources": True,
            "should_acknowledge_limitations": True,
            "expected_outcome": "both policies share similar exclusions"
        }
    },
    {
        "name": "Comparison - Price Differences",
        "query": "What are the price differences between SingLife Essential Critical Illness II and FWD Critical Illness Plus?",
        "expected": {
            "must_contain": ["premium", "singlife", "fwd"],
            "must_cite_both_policies": True,
            "must_cite_sources": True,
            "should_not_claim": ["cheaper", "more expensive"],
            "must_acknowledge_different_profiles": True
        }
    },
    {
        "name": "Comparison - Pros and Cons",
        "query": "What are the pros and cons of selecting SingLife Essential Critical Illness II over FWD Critical Illness Plus?",
        "expected": {
            "must_contain": ["singlife", "fwd", "pros", "cons"],
            "must_cite_both_policies": True,
            "must_cite_sources": True,
            "should_not_claim": ["always better", "never choose"]
        }
    },
    {
        "name": "Comparison - Coverage Differences",
        "query": "What does the critical illness policy from FWD cover that SingLife's one does not?",
        "expected": {
            "must_contain": ["fwd", "singlife", "cover"],
            "must_cite_both_policies": True,
            "must_cite_sources": True,
            "should_acknowledge_limitations": True
        }
    },
    {
        "name": "Recommendation - Main Reason",
        "query": "What is the main reason someone should select SingLife Essential Critical Illness II over FWD Critical Illness Plus?",
        "expected": {
            "must_contain": ["singlife", "pre-existing"],
            "must_cite_sources": True,
            "expected_outcome": "pre-existing conditions"
        }
    },
    {
        "name": "Single Policy - Coverage Details",
        "query": "What medical conditions are covered under SingLife Essential Critical Illness II?",
        "expected": {
            "must_contain": ["singlife", "condition", "cover"],
            "must_cite_sources": True
        }
    },
    {
        "name": "Single Policy - Claim Process",
        "query": "How do I file a claim with FWD Critical Illness Plus?",
        "expected": {
            "must_contain": ["fwd", "claim"],
            "must_cite_sources": True
        }
    },
    {
        "name": "Comparison - Claim Approval Rates",
        "query": "What are the claim approval rates for SingLife vs FWD?",
        "expected": {
            "must_cite_sources": True,
            "should_acknowledge_limitations": True,
            "expected_outcome": "don't provide information about claim approval rates"
        }
    }
]


def run_evaluation(batch_name: str = "insurance"):
    """Run all test cases and generate evaluation report."""

    print("="*80)
    print("INSURANCE POLICY RAG EVALUATION")
    print("="*80)
    print()

    # Initialize components
    batch_manager = BatchManager()
    query_processor = QueryProcessor(batch_manager)
    evaluator = QueryEvaluator(batch_manager, query_processor)

    # Switch to test batch
    if not batch_manager.switch_batch(batch_name):
        print(f"Failed to load batch '{batch_name}'")
        return

    # Run tests
    results = []
    passed_tests = 0
    total_tests = len(TEST_CASES)

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{total_tests}] {test_case['name']}")
        print("-" * 80)
        print(f"Query: {test_case['query']}")
        print()

        # Get response
        response = query_processor.process_query(test_case['query'])

        print(f"Response:\n{response[:300]}...")
        print()

        # Evaluate
        eval_result = evaluator.evaluate_response(
            test_case['query'],
            response,
            test_case['expected']
        )

        results.append(eval_result)

        # Print results
        if eval_result['passed']:
            passed_tests += 1
            print(f"✅ PASSED ({eval_result['score']}/{eval_result['max_score']} checks)")
        else:
            print(f"❌ FAILED ({eval_result['score']}/{eval_result['max_score']} checks)")

        if eval_result['failures']:
            print("Failures:")
            for failure in eval_result['failures']:
                print(f"  - {failure}")

        if eval_result['warnings']:
            print("Warnings:")
            for warning in eval_result['warnings']:
                print(f"  - {warning}")

        print()

    # Summary
    print("="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Tests Passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
    print()

    # Detailed breakdown
    total_score = sum(r['score'] for r in results)
    total_max_score = sum(r['max_score'] for r in results)
    print(f"Overall Score: {total_score}/{total_max_score} ({(total_score/total_max_score)*100:.1f}%)")
    print()

    # Save results
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': (passed_tests/total_tests)*100,
                'overall_score': total_score,
                'max_score': total_max_score,
                'score_percentage': (total_score/total_max_score)*100
            },
            'results': results
        }, f, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation tests")
    parser.add_argument("--batch", default="insurance", help="Batch name to test against")

    args = parser.parse_args()

    run_evaluation(args.batch)