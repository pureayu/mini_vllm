from __future__ import annotations

from collections import deque

from mini_vllm.kv_cache_manager import KVCacheManager
from mini_vllm.request import Request, RequestStatus


class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager, max_batch_size: int) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.max_batch_size = max_batch_size
        self.waiting_queue = deque()
        self.running_requests: list[Request] = []
        self.finished_requests: list[Request] = []
        self.step_id = 0

    #当有新的任务，调度器先将任务放到等待队列，等待后续准入
    def add_request(self, request: Request) -> None:
        self.waiting_queue.append(request)

    #准入后，等待队列的任务放到running队列，同时分配资源
    def admit_requests(self) -> None:
        while len(self.running_requests) < self.max_batch_size and self.waiting_queue:
            req = self.waiting_queue.popleft()
            req.status = RequestStatus.RUNNING
            self.running_requests.append(req)
            #准入控制，放入running队列中才能分配资源
            self.kv_cache_manager.register_request(req)


    #相当于开始运行了
    def prefill_request(self, request: Request) -> None:
        #先分配block
        for token_index in range(len(request.prompt_tokens)):
            self.kv_cache_manager.ensure_capacity(request, token_index=token_index)
        request.num_computed_tokens = len(request.prompt_tokens)
        request.prefill_done = True
        
        

    def decode_request(self, request: Request) -> None:
        if request.is_finished():
            return

        next_token = request.total_tokens() + 100
        self.kv_cache_manager.ensure_capacity(request, request.total_tokens())
        request.generated_tokens.append(next_token)
        request.num_computed_tokens += 1
        


    def finish_request(self, request: Request) -> None:
        self.running_requests.remove(request)
        self.kv_cache_manager.free_request(request.request_id)
        request.status = RequestStatus.FINISHED
        self.finished_requests.append(request)
        

    #每次调用step就代表系统往前推进一轮
    def step(self) -> None:
        #先将wait的请求准入
        if self.waiting_queue:
            self.admit_requests()
        #先prefill
        for req in list(self.running_requests):
            if req.prefill_done == False:
                self.prefill_request(req)
            else:
                self.decode_request(req)
            if req.is_finished():
                self.finish_request(req)
        self.step_id += 1
        


