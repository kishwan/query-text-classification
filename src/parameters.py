from dataclasses import dataclass

@dataclass
class parameters:
   in_channels: int = 35
   num_words: int = 2000
   
   embedding_size: int = 64
   out_channels: int = 32
   stride: int = 2
   
   epochs: int = 10
   batch_size: int = 12
   learning_rate: float = 0.001