from __future__ import annotations

from collections import deque

from mini_vllm.block_table import BlockTable
from mini_vllm.request import Request

class KVCacheManager:
    #初始化，在启动阶段，由engine启动并传入
    #属于系统配置层面的
    def __init__(self, num_blocks:int, block_size:int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        #关键变量，维护一个二维的BlockTable表
        #实现从请求到BlockTable的映射
        self.block_tables : dict[int, BlockTable] = {}
        #双端队列，维护空闲block
        self.free_block_ids = deque(range(num_blocks))

    def num_free_blocks(self)->int:
        return len(self.free_block_ids)

    #获取当前请求对应的BlockTable
    def get_block_table(self, request_id:int) -> BlockTable:
        return self.block_tables[request_id]
    
    #注册新的请求
    def register_request(self, request: Request) -> None:
        #创建新的BlockTable，相当于只是有个名字，但是还没有空间
        self.block_tables[request.request_id] = BlockTable(request.request_id, 
                                block_size=self.block_size)
    
    #重点：由于是按照step推进，因此并不是上来就分配了很大的空间
    #这个函数只有在block满了的情况下才会分配一个
    def allocate_block(self, request_id: int) -> int:
        if not self.free_block_ids:
            raise RuntimeError("No free KV cache blocks available.")
        
        free_block_id = self.free_block_ids.popleft()
        #先取出BlockTable对象，然后调用对应的append方法
        self.block_tables[request_id].append_block(free_block_id)

        return free_block_id
    
    #增加新token的时候判断是否需要开创新block
    #写 KV 之前，先问一下：容量够不够？不够就扩一块。
    def ensure_capacity(self, request: Request, token_index: int) -> None:
        if not self.block_tables[request.request_id].has_capacity_for(token_index):
            self.allocate_block(request.request_id)
        
    #清空请求对应的block
    #这对应着请求结束？
    def free_request(self, request_id: int) -> list[int]:
        block_table = self.block_tables.pop(request_id)
        freed_blocks = block_table.pop_all_blocks()

        for block_id in freed_blocks:
            self.free_block_ids.append(block_id)
        return freed_blocks
        
