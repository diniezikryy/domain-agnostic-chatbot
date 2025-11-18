"""
Script to verify ground truths by searching policy documents
"""
from query_processor import QueryProcessor
from batch_manager import BatchManager
import json

def search_and_display(qp, query, label):
    """Search and display top results"""
    print(f"\n{'='*80}")
    print(f"SEARCHING: {label}")
    print(f"Query: {query}")
    print('='*80)
    
    results = qp.search_engine.hybrid_search(query, top_k=3)
    
    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {})
        print(f"\n--- Result {i} ---")
        print(f"File: {metadata.get('filename', 'Unknown')}")
        print(f"Page: {metadata.get('page_number', 'N/A')}")
        print(f"Heading: {metadata.get('page_heading', 'N/A')}")
        print(f"Content preview: {r.get('content', '')[:400]}...")
        print()
    
    return results

def main():
    batch_mgr = BatchManager()
    qp = QueryProcessor(batch_mgr)
    
    # Ensure batch is loaded
    if not qp._ensure_batch_loaded("my_policies"):
        print("Failed to load batch!")
        return
    
    print("Batch loaded successfully!")
    
    # AUTO_013: Pre-existing condition exclusions
    search_and_display(qp, 
        "pre-existing condition exclusion critical illness before issue date",
        "AUTO_013 - Pre-existing condition exclusion (Manulife)")
    
    search_and_display(qp,
        "pre-existing conditions not covered unless disclosed accepted SupremeHealth",
        "AUTO_013 - Pre-existing condition (SupremeHealth)")
    
    # AUTO_014: Major Burns and Accidental Dental
    search_and_display(qp,
        "Major Burns critical illness 36 conditions",
        "AUTO_014 - Major Burns critical illness")
    
    search_and_display(qp,
        "Accidental Dental Treatment P PLUS",
        "AUTO_014 - Accidental Dental coverage")
    
    # AUTO_015: Flight delay
    search_and_display(qp,
        "travel delay overseas 4 hours 100 TravelCare",
        "AUTO_015 - Travel delay benefit")
    
    # AUTO_016: Cancer recurrence
    search_and_display(qp,
        "critical illness exclusion condition existed before supplementary benefit issue",
        "AUTO_016 - Cancer recurrence exclusion")
    
    # AUTO_017: Lifetime cancer coverage
    search_and_display(qp,
        "lifetime benefit limit unlimited cancer treatment SupremeHealth",
        "AUTO_017 - Lifetime cancer coverage")
    
    search_and_display(qp,
        "outpatient cancer drug treatment multiple primary cancers",
        "AUTO_017 - Cancer drug coverage")
    
    # AUTO_018: Overseas outpatient
    search_and_display(qp,
        "emergency medical treatment outside Singapore overseas",
        "AUTO_018 - Overseas emergency coverage (SupremeHealth)")
    
    search_and_display(qp,
        "pre-existing conditions overseas medical expenses TravelCare",
        "AUTO_018 - Pre-existing overseas coverage (TravelCare)")

if __name__ == "__main__":
    main()
