import sys
import os

# Ensure repository root is on sys.path so imports like `utils.*` resolve
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.cache_manager import CacheManager


if __name__ == '__main__':
    cm = CacheManager()
    cm.clear(experiment_name='combined_best', batch_id='my_policies')
    print('Done clearing combined_best cache for batch my_policies')
