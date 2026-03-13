from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_vllm.kv_cache_manager import KVCacheManager
from mini_vllm.request import Request
from mini_vllm.scheduler import Scheduler


def main() -> None:
    kv_cache_manager = KVCacheManager(num_blocks=8, block_size=4)
    scheduler = Scheduler(kv_cache_manager=kv_cache_manager, max_batch_size=2)

    request_1 = Request(
        request_id=1,
        max_tokens=2,
        num_computed_tokens=0,
        prompt_tokens=[10, 11, 12],
    )
    request_2 = Request(
        request_id=2,
        max_tokens=3,
        num_computed_tokens=0,
        prompt_tokens=[20, 21],
    )

    scheduler.add_request(request_1)
    scheduler.add_request(request_2)

    print("before step:")
    print("waiting:", len(scheduler.waiting_queue))
    print("running:", len(scheduler.running_requests))
    print("finished:", len(scheduler.finished_requests))
    print("free blocks:", kv_cache_manager.num_free_blocks())
    print()

    for i in range(5):
        scheduler.step()
        print(f"after step {i + 1}:")
        print("waiting:", len(scheduler.waiting_queue))
        print("running:", len(scheduler.running_requests))
        print("finished:", len(scheduler.finished_requests))
        print("free blocks:", kv_cache_manager.num_free_blocks())

        for req in scheduler.finished_requests:
            print(
                f"finished request {req.request_id}: "
                f"prefill_done={req.prefill_done}, "
                f"generated_tokens={req.generated_tokens}, "
                f"status={req.status}"
            )

        for req in scheduler.running_requests:
            print(
                f"running request {req.request_id}: "
                f"prefill_done={req.prefill_done}, "
                f"generated_tokens={req.generated_tokens}, "
                f"status={req.status}"
            )

        print()


if __name__ == "__main__":
    main()
