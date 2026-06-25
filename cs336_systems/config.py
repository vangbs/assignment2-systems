from dataclasses import dataclass

@dataclass(frozen=True)
class TilingConfig:
    block_q: int = 16
    block_k: int = 16