"""
Comparison Report Generator
Generates human-readable reports and visualizations from comparison results.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ComparisonReportGenerator:
    """
    Generates detailed reports from RAG vs Baseline comparison results.
    Produces markdown reports, summary statistics, and insights.
    """
    
    def __init__(self, results_file: str):
        """
        Initialize report generator.
        
        Args:
            results_file: Path to comparison results JSON file
        """
        self.results_file = results_file
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get('metadata', {})
        self.results = self.data.get('results', [])
    
    def generate_markdown_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive markdown report.
        
        Args:
            output_file: Optional path to save report (auto-generated if None)
            
        Returns:
            Path to saved report
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation/results/comparison_report_{timestamp}.md"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report content
        report = self._build_markdown_content()
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Markdown report saved to: {output_file}")
        return output_file
    
    def _build_markdown_content(self) -> str:
        """Build complete markdown report content."""
        sections = [
            self._header_section(),
            self._executive_summary_section(),
            self._performance_section(),
            self._quality_section(),
            self._citations_section(),
            self._individual_results_section(),
            self._recommendations_section()
        ]
        
        return "\n\n".join(sections)
    
    def _header_section(self) -> str:
        """Generate report header."""
        return f"""# RAG vs Baseline LLM Comparison Report

**Generated:** {self.metadata.get('timestamp', 'N/A')}  
**Batch:** {self.metadata.get('batch_name', 'N/A')}  
**Baseline Model:** {self.metadata.get('baseline_model', 'N/A')}  
**Total Queries:** {self.metadata.get('total_queries', 0)}

---"""
    
    def _executive_summary_section(self) -> str:
        """Generate executive summary."""
        total = len(self.results)
        
        # Calculate key metrics
        rag_quality_wins = sum(
            1 for r in self.results 
            if r['quality_comparison']['quality_scores']['rag_overall'] > 
               r['quality_comparison']['quality_scores']['baseline_overall']
        )
        
        avg_quality_improvement = sum(
            r['quality_comparison']['quality_scores']['rag_improvement'] 
            for r in self.results
        ) / total if total > 0 else 0
        
        avg_time_overhead = sum(
            r['performance_comparison']['time_difference_seconds'] 
            for r in self.results
        ) / total if total > 0 else 0
        
        total_rag_citations = sum(
            r['quality_comparison']['rag_quality']['citations_count'] 
            for r in self.results
        )
        
        total_baseline_citations = sum(
            r['quality_comparison']['baseline_quality']['citations_count'] 
            for r in self.results
        )
        
        return f"""## Executive Summary

### Key Findings

- **Quality Winner:** {'✅ RAG Pipeline' if rag_quality_wins > total/2 else '⚠️ Baseline LLM'}
- **RAG Win Rate:** {rag_quality_wins}/{total} ({rag_quality_wins/total*100:.1f}%)
- **Average Quality Improvement:** {avg_quality_improvement:+.3f} ({avg_quality_improvement*100:+.1f}%)
- **Average Time Overhead:** +{avg_time_overhead:.2f}s per query
- **Citation Improvement:** +{total_rag_citations - total_baseline_citations} citations ({total_rag_citations} vs {total_baseline_citations})

### Verdict

{'✅ **RAG pipeline provides superior quality** despite time overhead' if rag_quality_wins > total/2 else '⚠️ **Consider RAG configuration** - baseline performing competitively'}"""
    
    def _performance_section(self) -> str:
        """Generate performance comparison section."""
        total = len(self.results)
        
        avg_baseline_time = sum(r['baseline']['time_seconds'] for r in self.results) / total
        avg_rag_time = sum(r['rag']['time_seconds'] for r in self.results) / total
        
        min_baseline = min(r['baseline']['time_seconds'] for r in self.results)
        max_baseline = max(r['baseline']['time_seconds'] for r in self.results)
        min_rag = min(r['rag']['time_seconds'] for r in self.results)
        max_rag = max(r['rag']['time_seconds'] for r in self.results)
        
        return f"""## Performance Comparison

| Metric | Baseline LLM | RAG Pipeline | Difference |
|--------|--------------|--------------|------------|
| **Average Response Time** | {avg_baseline_time:.2f}s | {avg_rag_time:.2f}s | +{avg_rag_time - avg_baseline_time:.2f}s |
| **Min Response Time** | {min_baseline:.2f}s | {min_rag:.2f}s | +{min_rag - min_baseline:.2f}s |
| **Max Response Time** | {max_baseline:.2f}s | {max_rag:.2f}s | +{max_rag - max_baseline:.2f}s |
| **Overhead %** | - | - | {(avg_rag_time - avg_baseline_time)/avg_baseline_time*100:.1f}% |

### Analysis

- RAG pipeline adds approximately **{avg_rag_time - avg_baseline_time:.2f}s overhead** per query
- This overhead is due to document retrieval and context processing
- Performance is **{'acceptable' if avg_rag_time < 30 else 'concerning'}** for production use"""
    
    def _quality_section(self) -> str:
        """Generate quality comparison section."""
        total = len(self.results)
        
        avg_baseline_quality = sum(
            r['quality_comparison']['quality_scores']['baseline_overall'] 
            for r in self.results
        ) / total
        
        avg_rag_quality = sum(
            r['quality_comparison']['quality_scores']['rag_overall'] 
            for r in self.results
        ) / total
        
        # Count advantages
        advantage_counts = {
            'more_citations': 0,
            'lower_hallucination_risk': 0,
            'more_comprehensive': 0,
            'better_structured': 0,
            'more_specific': 0
        }
        
        for r in self.results:
            for advantage, has_it in r['quality_comparison']['rag_advantages'].items():
                if has_it:
                    advantage_counts[advantage] += 1
        
        return f"""## Quality Comparison

| Metric | Baseline LLM | RAG Pipeline | Winner |
|--------|--------------|--------------|--------|
| **Average Quality Score** | {avg_baseline_quality:.3f} | {avg_rag_quality:.3f} | {'🏆 RAG' if avg_rag_quality > avg_baseline_quality else '🏆 Baseline'} |
| **Citation Score** | {sum(r['quality_comparison']['baseline_quality']['citation_score'] for r in self.results)/total:.3f} | {sum(r['quality_comparison']['rag_quality']['citation_score'] for r in self.results)/total:.3f} | {'🏆 RAG' if sum(r['quality_comparison']['rag_quality']['citation_score'] for r in self.results) > sum(r['quality_comparison']['baseline_quality']['citation_score'] for r in self.results) else '🏆 Baseline'} |
| **Hallucination Risk** | {sum(r['quality_comparison']['baseline_quality']['hallucination_risk_score'] for r in self.results)/total:.3f} | {sum(r['quality_comparison']['rag_quality']['hallucination_risk_score'] for r in self.results)/total:.3f} | {'🏆 RAG' if sum(r['quality_comparison']['rag_quality']['hallucination_risk_score'] for r in self.results) < sum(r['quality_comparison']['baseline_quality']['hallucination_risk_score'] for r in self.results) else '🏆 Baseline'} (lower=better) |
| **Comprehensiveness** | {sum(r['quality_comparison']['baseline_quality']['comprehensiveness_score'] for r in self.results)/total:.3f} | {sum(r['quality_comparison']['rag_quality']['comprehensiveness_score'] for r in self.results)/total:.3f} | {'🏆 RAG' if sum(r['quality_comparison']['rag_quality']['comprehensiveness_score'] for r in self.results) > sum(r['quality_comparison']['baseline_quality']['comprehensiveness_score'] for r in self.results) else '🏆 Baseline'} |

### RAG Advantages (Frequency)

| Advantage | Frequency |
|-----------|-----------|
| More Citations | {advantage_counts['more_citations']}/{total} ({advantage_counts['more_citations']/total*100:.1f}%) |
| Lower Hallucination Risk | {advantage_counts['lower_hallucination_risk']}/{total} ({advantage_counts['lower_hallucination_risk']/total*100:.1f}%) |
| More Comprehensive | {advantage_counts['more_comprehensive']}/{total} ({advantage_counts['more_comprehensive']/total*100:.1f}%) |
| Better Structured | {advantage_counts['better_structured']}/{total} ({advantage_counts['better_structured']/total*100:.1f}%) |
| More Specific | {advantage_counts['more_specific']}/{total} ({advantage_counts['more_specific']/total*100:.1f}%) |"""
    
    def _citations_section(self) -> str:
        """Generate citations analysis section."""
        total = len(self.results)
        
        total_baseline_citations = sum(
            r['quality_comparison']['baseline_quality']['citations_count'] 
            for r in self.results
        )
        
        total_rag_citations = sum(
            r['quality_comparison']['rag_quality']['citations_count'] 
            for r in self.results
        )
        
        baseline_with_citations = sum(
            1 for r in self.results 
            if r['quality_comparison']['baseline_quality']['citations_count'] > 0
        )
        
        rag_with_citations = sum(
            1 for r in self.results 
            if r['quality_comparison']['rag_quality']['citations_count'] > 0
        )
        
        return f"""## Citation Analysis

| Metric | Baseline LLM | RAG Pipeline |
|--------|--------------|--------------|
| **Total Citations** | {total_baseline_citations} | {total_rag_citations} |
| **Avg per Response** | {total_baseline_citations/total:.2f} | {total_rag_citations/total:.2f} |
| **Responses w/ Citations** | {baseline_with_citations}/{total} ({baseline_with_citations/total*100:.1f}%) | {rag_with_citations}/{total} ({rag_with_citations/total*100:.1f}%) |
| **Citation Density** | {sum(r['quality_comparison']['baseline_quality']['citation_density'] for r in self.results)/total:.2f}/100 words | {sum(r['quality_comparison']['rag_quality']['citation_density'] for r in self.results)/total:.2f}/100 words |

### Key Insight

{'✅ RAG provides **significantly more citations**, improving trustworthiness and verifiability' if total_rag_citations > total_baseline_citations * 2 else '⚠️ Citation improvement exists but may need tuning'}"""
    
    def _individual_results_section(self) -> str:
        """Generate individual query results section."""
        lines = ["## Individual Query Results\n"]
        
        for i, result in enumerate(self.results, 1):
            query_id = result.get('query_id', f'query_{i}')
            query = result['query']
            
            baseline_time = result['baseline']['time_seconds']
            rag_time = result['rag']['time_seconds']
            
            baseline_quality = result['quality_comparison']['quality_scores']['baseline_overall']
            rag_quality = result['quality_comparison']['quality_scores']['rag_overall']
            
            winner = "🏆 RAG" if rag_quality > baseline_quality else "🏆 Baseline"
            
            lines.append(f"### {i}. {query_id}")
            lines.append(f"**Query:** {query}\n")
            lines.append(f"| Metric | Baseline | RAG | Winner |")
            lines.append(f"|--------|----------|-----|--------|")
            lines.append(f"| Time | {baseline_time:.2f}s | {rag_time:.2f}s | {'Baseline' if baseline_time < rag_time else 'RAG'} |")
            lines.append(f"| Quality | {baseline_quality:.3f} | {rag_quality:.3f} | {winner} |")
            lines.append(f"| Citations | {result['quality_comparison']['baseline_quality']['citations_count']} | {result['quality_comparison']['rag_quality']['citations_count']} | {'RAG' if result['quality_comparison']['rag_quality']['citations_count'] > result['quality_comparison']['baseline_quality']['citations_count'] else 'Baseline'} |")
            
            # Add side-by-side response comparison
            lines.append("\n#### Response Comparison\n")
            lines.append("<table>")
            lines.append("<tr>")
            lines.append("<th width='50%'>Baseline LLM Response</th>")
            lines.append("<th width='50%'>RAG Pipeline Response</th>")
            lines.append("</tr>")
            lines.append("<tr>")
            lines.append("<td valign='top'>")
            lines.append("")
            # Format baseline response
            baseline_response = result['baseline']['response'].strip()
            lines.append(baseline_response)
            lines.append("")
            lines.append("</td>")
            lines.append("<td valign='top'>")
            lines.append("")
            # Format RAG response
            rag_response = result['rag']['response'].strip()
            lines.append(rag_response)
            lines.append("")
            lines.append("</td>")
            lines.append("</tr>")
            lines.append("</table>")
            lines.append("")
        
        return "\n".join(lines)
    
    def _recommendations_section(self) -> str:
        """Generate recommendations section."""
        total = len(self.results)
        
        rag_quality_wins = sum(
            1 for r in self.results 
            if r['quality_comparison']['quality_scores']['rag_overall'] > 
               r['quality_comparison']['quality_scores']['baseline_overall']
        )
        
        avg_time_overhead = sum(
            r['performance_comparison']['time_difference_seconds'] 
            for r in self.results
        ) / total
        
        avg_hallucination_improvement = sum(
            r['quality_comparison']['baseline_quality']['hallucination_risk_score'] -
            r['quality_comparison']['rag_quality']['hallucination_risk_score']
            for r in self.results
        ) / total
        
        recommendations = ["## Recommendations\n"]
        
        if rag_quality_wins > total * 0.8:
            recommendations.append("✅ **Deploy RAG Pipeline** - Clear quality advantage across most queries")
        elif rag_quality_wins > total * 0.5:
            recommendations.append("✅ **Deploy RAG with monitoring** - Quality advantage exists, monitor edge cases")
        else:
            recommendations.append("⚠️ **Review RAG configuration** - Baseline competitive, investigate specific failures")
        
        if avg_time_overhead > 20:
            recommendations.append("\n⚠️ **Optimize performance** - Response time overhead exceeds 20s")
            recommendations.append("- Consider caching strategies")
            recommendations.append("- Review retrieval K parameters")
            recommendations.append("- Optimize embedding generation")
        
        if avg_hallucination_improvement > 0.1:
            recommendations.append("\n✅ **Hallucination reduction validated** - RAG significantly reduces hallucination risk")
        
        return "\n".join(recommendations)
    
    def print_summary(self):
        """Print concise summary to console."""
        total = len(self.results)
        
        rag_wins = sum(
            1 for r in self.results 
            if r['quality_comparison']['quality_scores']['rag_overall'] > 
               r['quality_comparison']['quality_scores']['baseline_overall']
        )
        
        avg_baseline_time = sum(r['baseline']['time_seconds'] for r in self.results) / total
        avg_rag_time = sum(r['rag']['time_seconds'] for r in self.results) / total
        
        print(f"\n{'='*80}")
        print("COMPARISON REPORT SUMMARY")
        print(f"{'='*80}")
        print(f"Source: {self.results_file}")
        print(f"Total Queries: {total}")
        print(f"Timestamp: {self.metadata.get('timestamp', 'N/A')}")
        
        print(f"\n📊 Quality Results:")
        print(f"  RAG Wins: {rag_wins}/{total} ({rag_wins/total*100:.1f}%)")
        print(f"  Baseline Wins: {total-rag_wins}/{total} ({(total-rag_wins)/total*100:.1f}%)")
        
        print(f"\n⏱️  Performance:")
        print(f"  Baseline avg: {avg_baseline_time:.2f}s")
        print(f"  RAG avg: {avg_rag_time:.2f}s")
        print(f"  Overhead: +{avg_rag_time - avg_baseline_time:.2f}s")
        
        print(f"\n{'='*80}\n")


def generate_report_from_file(results_file: str, output_file: Optional[str] = None):
    """
    Generate report from comparison results file.
    
    Args:
        results_file: Path to comparison results JSON
        output_file: Optional output path for markdown report
    """
    generator = ComparisonReportGenerator(results_file)
    generator.print_summary()
    markdown_file = generator.generate_markdown_report(output_file)
    
    print(f"✅ Report generation complete")
    print(f"📄 Markdown report: {markdown_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comparison_report_generator.py <results_file.json> [output_file.md]")
        print("\nExample:")
        print("  python evaluation/comparison_report_generator.py evaluation/results/rag_vs_baseline_comparison_20251029_143000.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_report_from_file(results_file, output_file)
