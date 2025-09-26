"""
Batch Manager
Handles switching between different document batches and managing batch registry.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

class BatchManager:
    def __init__(self, batches_dir: str = "batches"):
        self.batches_dir = Path(batches_dir)
        self.batches_dir.mkdir(exist_ok=True)
        self.registry_file = self.batches_dir / "batch_registry.json"
        self.current_batch = None
        self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load batch registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # Create empty registry
        return {
            "batches": {},
            "default_batch": None,
            "last_modified": datetime.now().isoformat()
        }

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save batch registry to file."""
        registry["last_modified"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def register_batch(self, batch_id: str, batch_info: Dict[str, Any]) -> bool:
        """Register a new batch in the registry."""
        try:
            registry = self._load_registry()
            registry["batches"][batch_id] = {
                "id": batch_id,
                "name": batch_info.get("name", batch_id),
                "description": batch_info.get("description", ""),
                "doc_count": batch_info.get("doc_count", 0),
                "created_at": batch_info.get("created_at", datetime.now().isoformat()),
                "faiss_path": f"batches/{batch_id}/faiss_index",
                "bm25_path": f"batches/{batch_id}/bm25_index.pkl",
                "metadata_path": f"batches/{batch_id}/metadata.json"
            }

            # Set as default if it's the first batch
            if not registry["default_batch"]:
                registry["default_batch"] = batch_id

            self._save_registry(registry)
            return True
        except Exception as e:
            print(f"Error registering batch: {e}")
            return False

    def list_batches(self) -> Dict[str, Any]:
        """List all available batches."""
        registry = self._load_registry()
        return registry.get("batches", {})

    def get_batch_info(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific batch."""
        registry = self._load_registry()
        return registry.get("batches", {}).get(batch_id)

    def switch_batch(self, batch_id: str) -> bool:
        """Switch to a specific batch."""
        batch_info = self.get_batch_info(batch_id)
        if not batch_info:
            print(f"Batch '{batch_id}' not found.")
            return False

        # Verify batch files exist
        batch_dir = self.batches_dir / batch_id
        faiss_dir = batch_dir / "faiss_index"
        bm25_file = batch_dir / "bm25_index.pkl"

        if not batch_dir.exists():
            print(f"Batch directory not found: {batch_dir}")
            return False

        if not faiss_dir.exists() or not bm25_file.exists():
            print(f"Batch files incomplete for '{batch_id}'")
            print(f"FAISS index: {'✓' if faiss_dir.exists() else '✗'}")
            print(f"BM25 index: {'✓' if bm25_file.exists() else '✗'}")
            return False

        self.current_batch = batch_id
        print(f"Switched to batch: {batch_id}")
        return True

    def get_current_batch(self) -> Optional[str]:
        """Get the currently active batch."""
        return self.current_batch

    def get_default_batch(self) -> Optional[str]:
        """Get the default batch."""
        registry = self._load_registry()
        return registry.get("default_batch")

    def set_default_batch(self, batch_id: str) -> bool:
        """Set the default batch."""
        if not self.get_batch_info(batch_id):
            return False

        registry = self._load_registry()
        registry["default_batch"] = batch_id
        self._save_registry(registry)
        return True

    def delete_batch(self, batch_id: str) -> bool:
        """Delete a batch and its files."""
        try:
            registry = self._load_registry()

            if batch_id not in registry.get("batches", {}):
                return False

            # Remove batch directory
            batch_dir = self.batches_dir / batch_id
            if batch_dir.exists():
                shutil.rmtree(batch_dir)

            # Remove from registry
            del registry["batches"][batch_id]

            # Update default batch if necessary
            if registry.get("default_batch") == batch_id:
                remaining_batches = list(registry["batches"].keys())
                registry["default_batch"] = remaining_batches[0] if remaining_batches else None

            self._save_registry(registry)
            return True

        except Exception as e:
            print(f"Error deleting batch: {e}")
            return False

    def get_batch_paths(self, batch_id: str) -> Optional[Dict[str, str]]:
        """Get file paths for a batch."""
        batch_info = self.get_batch_info(batch_id)
        if not batch_info:
            return None

        batch_dir = self.batches_dir / batch_id
        return {
            "faiss_index": str(batch_dir / "faiss_index"),
            "bm25_index": str(batch_dir / "bm25_index.pkl"),
            "metadata": str(batch_dir / "metadata.json")
        }