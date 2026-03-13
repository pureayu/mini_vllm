from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vllm.kv_cache_manager import KVCacheManager
from mini_vllm.request import Request


def main() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=4)

    request = Request(
        request_id=1,
        max_tokens=6,
        num_computed_tokens=0,
        prompt_tokens=[10, 11, 12],
    )

    manager.register_request(request)

    print("initial free blocks:", manager.num_free_blocks())
    print("initial block table:", manager.get_block_table(request.request_id).physical_block_ids)

    manager.ensure_capacity(request, token_index=0)
    print("after token 0:", manager.get_block_table(request.request_id).physical_block_ids)

    manager.ensure_capacity(request, token_index=3)
    print("after token 3:", manager.get_block_table(request.request_id).physical_block_ids)

    manager.ensure_capacity(request, token_index=4)
    print("after token 4:", manager.get_block_table(request.request_id).physical_block_ids)

    freed_blocks = manager.free_request(request.request_id)
    print("freed blocks:", freed_blocks)
    print("free blocks after reclaim:", manager.num_free_blocks())


if __name__ == "__main__":
    main()
