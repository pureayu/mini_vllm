from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    
@dataclass
class Request:
    request_id : int
    #请求的最大token数量
    max_tokens : int
    #已经完成的token数量
    num_computed_tokens : int

    prompt_tokens : list[int]
    prefill_done : bool = False
    generated_tokens : list[int] = field(default_factory=list)
    status : RequestStatus = RequestStatus.WAITING
    
    
    def computed_tokens(self):
        return self.num_computed_tokens
    
    def is_finished(self):
        return len(self.generated_tokens) >= self.max_tokens
    
    def total_tokens(self):
        return len(self.prompt_tokens) + len(self.generated_tokens)
    
    
