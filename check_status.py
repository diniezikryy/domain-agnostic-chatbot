"""
Project Status Checker
Verifies setup and identifies missing components.
"""

import os
import sys
from pathlib import Path
import json

def check_status():
    """Check project setup status."""
    print("\n" + "="*80)
    print("PROJECT SETUP STATUS CHECK")
    print("="*80)
    
    issues = []
    warnings = []
    
    # Check 1: API Key
    print("\n1. API Key Configuration")
    try:
        from config.settings import Settings
        app_settings = Settings()
        if app_settings.openai_api_key:
            print("   ✓ API key is configured")
        else:
            print("   ✗ API key NOT set")
            issues.append("Add OPENAI_API_KEY to .env file")
    except Exception as e:
        print(f"   ✗ Error checking API key: {e}")
        issues.append("Fix config/settings.py or .env file")
    
    # Check 2: Documents Directory
    print("\n2. Documents Directory")
    docs_dir = Path("documents")
    if docs_dir.exists():
        print("   ✓ documents/ directory exists")
        
        # Check subdirectories
        subdirs = [d for d in docs_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"   ✓ Found {len(subdirs)} subdirectories:")
            for subdir in subdirs:
                files = list(subdir.glob("*.pdf")) + list(subdir.glob("*.docx")) + \
                       list(subdir.glob("*.txt")) + list(subdir.glob("*.md"))
                if files:
                    print(f"     • {subdir.name}: {len(files)} documents")
                else:
                    print(f"     ⚠ {subdir.name}: NO DOCUMENTS")
                    warnings.append(f"Add documents to documents/{subdir.name}/")
        else:
            print("   ⚠ No subdirectories found")
            warnings.append("Create subdirectories in documents/ for your batches")
    else:
        print("   ✗ documents/ directory does NOT exist")
        issues.append("Create documents/ directory and add subdirectories")
    
    # Check 3: Batches Directory
    print("\n3. Batches Directory")
    batches_dir = Path("batches")
    if batches_dir.exists():
        print("   ✓ batches/ directory exists")
        
        # Check registry
        registry_file = batches_dir / "batch_registry.json"
        if registry_file.exists():
            print("   ✓ batch_registry.json exists")
            
            try:
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                
                batches = registry.get('batches', {})
                if batches:
                    print(f"   ✓ Registry shows {len(batches)} batches:")
                    
                    for batch_id, batch_info in batches.items():
                        batch_dir = Path(batch_info.get('faiss_path', '')).parent
                        if batch_dir.exists():
                            # Check for required files
                            has_faiss = (batch_dir / "faiss_index").exists() or \
                                       any(batch_dir.glob("faiss_index.*"))
                            has_bm25 = (batch_dir / "bm25_index.pkl").exists()
                            has_metadata = (batch_dir / "metadata.json").exists()
                            
                            if has_faiss and has_bm25 and has_metadata:
                                print(f"     ✓ {batch_id}: READY")
                            else:
                                print(f"     ✗ {batch_id}: INCOMPLETE")
                                missing = []
                                if not has_faiss: missing.append("faiss_index")
                                if not has_bm25: missing.append("bm25_index.pkl")
                                if not has_metadata: missing.append("metadata.json")
                                print(f"       Missing: {', '.join(missing)}")
                                issues.append(f"Rebuild batch '{batch_id}': python setup_batch.py {batch_id}")
                        else:
                            print(f"     ✗ {batch_id}: DIRECTORY MISSING")
                            issues.append(f"Rebuild batch '{batch_id}': python setup_batch.py {batch_id}")
                else:
                    print("   ⚠ No batches in registry")
                    warnings.append("Create batches using: python setup_batch.py <batch_name>")
                    
            except Exception as e:
                print(f"   ✗ Error reading registry: {e}")
                issues.append("Fix or rebuild batch_registry.json")
        else:
            print("   ✗ batch_registry.json NOT found")
            issues.append("Create batches using: python setup_batch.py <batch_name>")
    else:
        print("   ✗ batches/ directory does NOT exist")
        issues.append("Create batches/ directory will be created automatically")
    
    # Check 4: Dependencies
    print("\n4. Python Dependencies")
    try:
        import openai
        print("   ✓ openai package installed")
    except ImportError:
        print("   ✗ openai package NOT installed")
        issues.append("Install dependencies: pip install -r requirements.txt")
    
    try:
        import faiss
        print("   ✓ faiss package installed")
    except ImportError:
        print("   ✗ faiss package NOT installed")
        issues.append("Install dependencies: pip install -r requirements.txt")
    
    try:
        import langchain
        print("   ✓ langchain package installed")
    except ImportError:
        print("   ✗ langchain package NOT installed")
        issues.append("Install dependencies: pip install -r requirements.txt")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if not issues and not warnings:
        print("\n✅ ALL CHECKS PASSED! Your project is ready to use.")
        print("\nNext steps:")
        print("  python main.py \"What is covered?\"")
        print("  python tests/test_rag_vs_baseline.py --batch <name> --max 3")
    else:
        if issues:
            print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\n📖 For detailed instructions, see: DOCUMENTS_SETUP.md")
    
    print("\n" + "="*80 + "\n")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = check_status()
    sys.exit(0 if success else 1)
