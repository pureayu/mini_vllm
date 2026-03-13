from __future__ import annotations

from mini_vllm.kv_cache_manager import KVCacheManager
from mini_vllm.request import Request
from mini_vllm.scheduler import Scheduler


class Engine:
    def __init__(self, num_blocks: int, block_size: int, max_batch_size: int) -> None:
        self.kv_cache_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_batch_size=max_batch_size,
        )

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def step(self) -> None:
        self.scheduler.step()

    def has_unfinished_requests(self) -> bool:
        return bool(self.scheduler.waiting_queue or self.scheduler.running_requests)

    def run(self) -> None:
        while self.has_unfinished_requests():
            self.step()

    def get_finished_requests(self) -> list[Request]:
        return self.scheduler.finished_requests
