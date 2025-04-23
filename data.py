import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Optional


class TextDataset(Dataset):
    """A simple dataset that loads text files for language modeling."""
    
    def __init__(self, file_path, tokenizer, seq_length=512, stride=256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride
        
        # Read input file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try again with a different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        # Tokenize the text
        self.tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        # Calculate the number of chunks
        self.num_chunks = max(1, (len(self.tokens) - seq_length) // stride + 1)
        
        # Precompute chunk boundaries for faster access
        self.chunk_boundaries = []
        for idx in range(self.num_chunks):
            start = idx * self.stride
            end = min(start + self.seq_length, len(self.tokens))
            
            # Handle the case where the chunk is shorter than seq_length
            if end - start < self.seq_length:
                start = max(0, end - self.seq_length)
            
            self.chunk_boundaries.append((start, end))
    
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        start, end = self.chunk_boundaries[idx]
        chunk = self.tokens[start:end].clone()  # Clone to avoid memory sharing issues
        
        # If the chunk is still smaller than seq_length, pad it
        if len(chunk) < self.seq_length:
            padding = torch.zeros(self.seq_length - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(chunk)
        attention_mask[len(self.tokens[start:end]):] = 0
        
        return {
            "input_ids": chunk,
            "attention_mask": attention_mask,
        }


class TextDataModule(pl.LightningDataModule):
    """Lightning Data Module for text training."""
    
    def __init__(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        tokenizer = None,
        batch_size: int = 4,  # Reduced batch size for CPU
        seq_length: int = 512,  # Reduced sequence length
        stride: int = 256,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file or train_file  # Default to train file if no val file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.stride = stride
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Setup datasets."""
        self.train_dataset = TextDataset(
            self.train_file,
            self.tokenizer,
            seq_length=self.seq_length,
            stride=self.stride,
        )
        
        self.val_dataset = TextDataset(
            self.val_file,
            self.tokenizer,
            seq_length=self.seq_length,
            stride=self.stride,
        )
    
    def train_dataloader(self):
        use_gpu = torch.cuda.is_available()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_gpu,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context='spawn' if use_gpu and self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        use_gpu = torch.cuda.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_gpu,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context='spawn' if use_gpu and self.num_workers > 0 else None,
        )


class StreamingTextDataset(Dataset):
    """A dataset that generates random data for testing or when actual data isn't needed."""
    
    def __init__(self, vocab_size=49152, seq_length=512, size=10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.size = size
        # Set default device for optimization
        self.device = 'cpu'  # Will be updated at access time if GPU is available
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Check if GPU is available at runtime to adapt to device availability
        if torch.cuda.is_available():
            # We still generate on CPU then transfer to avoid CUDA synchronization overhead
            # for each batch item generation
            device = 'cpu'
        else:
            device = 'cpu'
            
        # Generate random token ids efficiently
        # Using torch.randint is faster than torch.random_ for this purpose
        input_ids = torch.randint(1, self.vocab_size, (self.seq_length,), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class StreamingTextDataModule(pl.LightningDataModule):
    """Lightning Data Module for streaming random text data."""
    
    def __init__(
        self,
        vocab_size=49152,
        batch_size=4,  # Reduced batch size for CPU
        seq_length=512,  # Reduced sequence length
        train_size=10000,
        val_size=1000,
        num_workers=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.train_size = train_size
        self.val_size = val_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Setup datasets."""
        self.train_dataset = StreamingTextDataset(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            size=self.train_size,
        )
        
        self.val_dataset = StreamingTextDataset(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            size=self.val_size,
        )
    
    def train_dataloader(self):
        use_gpu = torch.cuda.is_available()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_gpu,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context='spawn' if use_gpu and self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        use_gpu = torch.cuda.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_gpu,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context='spawn' if use_gpu and self.num_workers > 0 else None,
        )