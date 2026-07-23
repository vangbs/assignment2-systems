from dataclasses import dataclass

@dataclass(frozen=True)
class TilingConfig:
    block_q: int = 32
    block_k: int = 32