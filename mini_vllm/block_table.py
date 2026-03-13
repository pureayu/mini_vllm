from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class BlockTable:
    request_id : int
    block_size : int
    physical_block_ids: list[int] = field(default_factory=list)

    def num_blocks(self):
        return len(self.physical_block_ids)
    
    def capacity_tokens(self):
        return self.block_size * len(self.physical_block_ids)
    
    #添加block
    def append_block(self, physical_block_id : int):
        self.physical_block_ids.append(physical_block_id)

    #根据逻辑块号映射到物理块号
    def physical_block_id(self, logical_block_id) -> int:
        return self.physical_block_ids[logical_block_id]
    
    #返回当前token在哪个逻辑块
    def logical_block_id(self, token_id : int) -> int:
        return token_id // self.block_size
    
    #判断当前request的block是否还有容量
    def has_capacity_for(self, token_index : int) -> bool:
        return self.capacity_tokens() > token_index
    
    def pop_all_blocks(self) -> list[int]:
        blocks = self.physical_block_ids.copy()
        self.physical_block_ids.clear()
        return blocks
