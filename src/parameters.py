from dataclasses import dataclass

@dataclass
class parameters:
   seq_len: int = 35
   num_words: int = 2000
   
   embedding_size: int = 64
   out_size: int = 32
   stride: int = 2
   
   epochs: int = 10
   batch_size: int = 12
   learning_rate: float = 0.001