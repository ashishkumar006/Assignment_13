import os
import argparse
import yaml
import torch
import numpy as np
import time
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
from model import SmolLM2LightningModule
from data import TextDataModule, StreamingTextDataModule


# Custom callback to track GPU usage and save metrics to TXT
class GPUStatsLogger(Callback):
    def __init__(self, output_dir):
        super().__init__()
        # Define paths for all log files
        self.gpu_logs_path = os.path.join(output_dir, "logs", "gpu_metrics.txt")
        self.loss_logs_path = os.path.join(output_dir, "logs", "training_loss.txt")
        self.epoch_logs_path = os.path.join(output_dir, "logs", "epoch_completion.txt")
        self.checkpoint_logs_path = os.path.join(output_dir, "logs", "checkpoint_events.txt")
        self.samples_path = os.path.join(output_dir, "logs", "generation_samples.txt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.gpu_logs_path), exist_ok=True)
        
        # Initialize the log files with headers
        with open(self.gpu_logs_path, 'w') as f:
            f.write("Timestamp | Step | GPU Utilization (%) | GPU Memory Used (MB) | GPU Memory Total (MB)\n")
            f.write("-" * 80 + "\n")
            
        with open(self.loss_logs_path, 'w') as f:
            f.write("Timestamp | Step | Train Loss | Validation Loss\n")
            f.write("-" * 50 + "\n")
            
        with open(self.epoch_logs_path, 'w') as f:
            f.write("Timestamp | Epoch | Duration (seconds) | Steps Completed\n")
            f.write("-" * 60 + "\n")
            
        with open(self.checkpoint_logs_path, 'w') as f:
            f.write("Timestamp | Event | Step | Details\n")
            f.write("-" * 50 + "\n")
            
        with open(self.samples_path, 'w') as f:
            f.write("SAMPLE GENERATIONS AT EVERY 500 STEPS\n")
            f.write("=" * 50 + "\n\n")
        
        # Track validation loss and epoch start time
        self.last_val_loss = float('nan')
        self.epoch_start_time = time.time()
        self.current_epoch = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return
            
        # Log every 10 steps to avoid excessive logging
        if trainer.global_step % 10 == 0:
            try:
                # Get current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Get GPU stats
                gpu_util = torch.cuda.utilization()
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
                
                # Get loss
                train_loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
                
                # Log to TXT files
                with open(self.gpu_logs_path, 'a') as f:
                    f.write(f"{timestamp} | {trainer.global_step} | {gpu_util} | {mem_allocated:.2f} | {mem_total:.2f}\n")
                    
                with open(self.loss_logs_path, 'a') as f:
                    f.write(f"{timestamp} | {trainer.global_step} | {train_loss:.6f} | {self.last_val_loss:.6f}\n")
                    
            except Exception as e:
                print(f"Error logging stats: {e}")
                
        # Log checkpoint events - find the checkpoint callback instead of assuming it's at index 0
        if trainer.global_step > 0:
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint) and hasattr(callback, 'every_n_train_steps'):
                    if trainer.global_step % callback.every_n_train_steps == 0:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(self.checkpoint_logs_path, 'a') as f:
                            f.write(f"{timestamp} | CHECKPOINT SAVED | {trainer.global_step} | Regular checkpoint\n")
                    break

    def on_validation_epoch_end(self, trainer, pl_module):
        if hasattr(trainer, 'callback_metrics') and 'val_loss' in trainer.callback_metrics:
            self.last_val_loss = trainer.callback_metrics['val_loss'].item()
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.current_epoch = trainer.current_epoch
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.epoch_logs_path, 'a') as f:
            f.write(f"{timestamp} | {self.current_epoch} | {epoch_duration:.2f} | {trainer.global_step}\n")

    def log_sample_generation(self, step, input_text, generated_text):
        """Log a sample generation to the samples file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.samples_path, 'a') as f:
            f.write(f"SAMPLE AT STEP {step} - {timestamp}\n")
            f.write("-" * 40 + "\n")
            f.write(f"INPUT: {input_text}\n\n")
            f.write(f"GENERATED OUTPUT:\n{generated_text}\n\n")
            f.write("=" * 40 + "\n\n")


def load_model_config(config_path):
    """Load model configuration from yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract model configuration
        model_config = config["model"]["model_config"]
        
        # Add optimizer config
        model_config["learning_rate"] = config["optimizer"]["learning_rate_scheduler"]["learning_rate"]
        model_config["weight_decay"] = config["optimizer"]["weight_decay"]
        model_config["adam_beta1"] = config["optimizer"]["optimizer_factory"]["adam_beta1"]
        model_config["adam_beta2"] = config["optimizer"]["optimizer_factory"]["adam_beta2"]
        model_config["adam_eps"] = config["optimizer"]["optimizer_factory"]["adam_eps"]
        
        # Add training config
        model_config["batch_size"] = config["tokens"]["micro_batch_size"]
        model_config["seq_length"] = config["tokens"]["sequence_length"]
        
        # Validate required model parameters
        required_params = [
            "hidden_size", "num_attention_heads", "num_key_value_heads", 
            "num_hidden_layers", "intermediate_size", "vocab_size",
            "max_position_embeddings", "rms_norm_eps", "rope_theta"
        ]
        
        for param in required_params:
            if param not in model_config:
                print(f"WARNING: Missing required parameter '{param}' in model config. Using default value.")
                # Set default values for missing parameters
                if param == "hidden_size":
                    model_config[param] = 576
                elif param == "num_attention_heads":
                    model_config[param] = 9
                elif param == "num_key_value_heads":
                    model_config[param] = 3
                elif param == "num_hidden_layers":
                    model_config[param] = 30
                elif param == "intermediate_size":
                    model_config[param] = 1536
                elif param == "vocab_size":
                    model_config[param] = 49152
                elif param == "max_position_embeddings":
                    model_config[param] = 2048
                elif param == "rms_norm_eps":
                    model_config[param] = 1e-5
                elif param == "rope_theta":
                    model_config[param] = 10000.0
        
        # Ensure num_heads is divisible by num_kv_heads
        if model_config["num_attention_heads"] % model_config["num_key_value_heads"] != 0:
            print(f"WARNING: num_attention_heads ({model_config['num_attention_heads']}) is not divisible by num_key_value_heads ({model_config['num_key_value_heads']}). Adjusting num_key_value_heads.")
            model_config["num_key_value_heads"] = 1
            
        # Ensure hidden_size is divisible by num_heads
        if model_config["hidden_size"] % model_config["num_attention_heads"] != 0:
            old_hidden_size = model_config["hidden_size"]
            model_config["hidden_size"] = ((model_config["hidden_size"] // model_config["num_attention_heads"]) + 1) * model_config["num_attention_heads"]
            print(f"WARNING: hidden_size ({old_hidden_size}) is not divisible by num_attention_heads ({model_config['num_attention_heads']}). Adjusting hidden_size to {model_config['hidden_size']}.")
            
        # Ensure required values
        model_config["hidden_act"] = model_config.get("hidden_act", "silu") 
        model_config["initializer_range"] = model_config.get("initializer_range", 0.042)
        model_config["tie_word_embeddings"] = model_config.get("tie_word_embeddings", True)
        model_config["use_cache"] = model_config.get("use_cache", True)
        
        return model_config
    except Exception as e:
        print(f"ERROR loading config: {e}")
        print("Using default model configuration")
        
        # Default model config for SmolLM2-135M
        return {
            "hidden_size": 576,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "num_hidden_layers": 30,
            "intermediate_size": 1536,
            "vocab_size": 49152,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "initializer_range": 0.042,
            "tie_word_embeddings": True,
            "use_cache": True,
            "batch_size": 4,
            "seq_length": 512,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1e-8,
        }


def main(args):
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Load model configuration
    config = load_model_config(args.config)
    
    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seq_length is not None:
        config["seq_length"] = args.seq_length
    
    # Load tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path}")
    else:
        # Use standard HF Llama tokenizer for testing
        try:
            tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
            print("Using default Llama tokenizer")
        except Exception as e:
            print(f"Error loading Llama tokenizer: {e}")
            print("Falling back to GPT-2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Optimize GPU memory usage if available
    if torch.cuda.is_available():
        print("GPU available. Optimizing memory usage.")
        # Empty GPU cache before starting
        torch.cuda.empty_cache()
        # Set memory allocation strategy to reduce fragmentation
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory to avoid OOM
    
    # Create data module with optimized settings for GPU
    if args.input_file:
        try:
            # For GPU, increase num_workers if available
            effective_workers = args.num_workers
            if torch.cuda.is_available() and args.num_workers == 0:
                effective_workers = min(4, os.cpu_count() or 1)  # Use 4 workers or number of CPUs, whichever is smaller
                
            data_module = TextDataModule(
                train_file=args.input_file,
                val_file=args.input_file,
                tokenizer=tokenizer,
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                num_workers=effective_workers,
            )
            print(f"Using text data from {args.input_file}")
            print(f"Batch size: {config['batch_size']}, Sequence length: {config['seq_length']}")
            print(f"Using {effective_workers} data loader workers")
        except Exception as e:
            print(f"Error loading text data: {e}")
            print("Falling back to streaming random data")
            data_module = StreamingTextDataModule(
                vocab_size=config["vocab_size"],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                num_workers=args.num_workers,
            )
    else:
        # Use streaming data for testing/demo
        data_module = StreamingTextDataModule(
            vocab_size=config["vocab_size"],
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            num_workers=args.num_workers,
        )
        print("Using streaming random data")
        print(f"Batch size: {config['batch_size']}, Sequence length: {config['seq_length']}")
    
    # Create model
    model = SmolLM2LightningModule(config, tokenizer)
    
    # Setup training callbacks with enhanced GPU monitoring
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="smollm2-135-{step:06d}",
            save_top_k=1,
            monitor="step",
            mode="max",
            every_n_train_steps=args.save_every,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        # Add TQDM Progress Bar with refresh rate
        TQDMProgressBar(refresh_rate=10),
        # Add custom GPU stats logger
        GPUStatsLogger(args.output_dir)
    ]
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name="smollm2",
    )
    
    # Configure trainer validation parameters
    validation_params = {}
    if args.validate_by_epoch:
        # Validate every epoch
        validation_params["check_val_every_n_epoch"] = 1
    else:
        # Validate every N steps (if specified)
        if args.val_check_interval:
            # Use a default value for validation interval (no longer trying to access dataloader)
            effective_val_interval = min(args.val_check_interval, 360)  # 360 as a safe default
            
            if effective_val_interval != args.val_check_interval:
                print(f"Warning: Validation interval adjusted from {args.val_check_interval} to {effective_val_interval} to match dataset size")
            
            validation_params["val_check_interval"] = effective_val_interval
            print(f"Will validate every {effective_val_interval} steps")
    
    # Configure enhanced precision settings for GPU
    precision = "32"
    if torch.cuda.is_available():
        # Check specifically for T4 GPU
        gpu_name = torch.cuda.get_device_name(0).lower()
        is_t4 = 't4' in gpu_name
        
        # Configure based on capability detection
        if torch.cuda.get_device_capability()[0] >= 8:
            # For Ampere (RTX 30xx+) or newer GPUs that support bfloat16 well
            try:
                # Check if bfloat16 is supported
                x = torch.tensor([1.0], device="cuda").to(torch.bfloat16)
                precision = "bf16-mixed"
                print("Using bfloat16 mixed precision training")
            except:
                precision = "16-mixed"
                print("Using float16 mixed precision training")
        else:
            # T4 and other Turing architecture GPUs work well with FP16
            try:
                x = torch.tensor([1.0], device="cuda").to(torch.float16)
                precision = "16-mixed"
                print(f"Using float16 mixed precision training{' optimized for T4 GPU' if is_t4 else ''}")
            except:
                # Fall back to float32
                precision = "32"
                print("Using float32 precision training")
    else:
        print("Using CPU with float32 precision")
    
    # Additional GPU optimizations
    strategy = "auto"
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = "ddp"  # Use distributed data parallel for multi-GPU
        print(f"Using {torch.cuda.device_count()} GPUs with DDP strategy")
    
    # Configure trainer with GPU optimizations
    try:
        trainer = pl.Trainer(
            max_steps=args.max_steps,
            accelerator="auto",
            devices="auto",
            precision=precision,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            strategy=strategy,
            accumulate_grad_batches=1,  # Can be increased for larger effective batch size
            **validation_params
        )
    except Exception as e:
        print(f"Error creating trainer with validation params: {e}")
        print("Falling back to basic trainer configuration")
        trainer = pl.Trainer(
            max_steps=args.max_steps,
            accelerator="auto",
            devices="auto",
            precision="32",
            callbacks=callbacks,
            logger=logger,
            check_val_every_n_epoch=1,  # Default to validate every epoch
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
        )
    
    # Train model
    print(f"Starting training for {args.max_steps} steps...")
    if args.validate_by_epoch:
        print("Validating every epoch")
    elif args.val_check_interval:
        print(f"Validating every {args.val_check_interval} steps")
    
    try:
        trainer.fit(model, data_module)
    except Exception as e:
        print(f"Error during training: {e}")
        print("Try reducing batch size or sequence length")
        return False
    
    # Free GPU memory before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, "checkpoints", "final_model.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")
    
    # Training complete
    print("Training complete!")
    
    # Continue training if requested
    if args.continue_steps > 0:
        print(f"Continuing training for {args.continue_steps} more steps from checkpoint...")
        
        try:
            # Load from checkpoint
            model = SmolLM2LightningModule.load_from_checkpoint(final_checkpoint_path)
            print(f"Loaded model from checkpoint: {final_checkpoint_path}")
            
            # Adjust learning rate for continuation to prevent NaN values
            if "learning_rate" in config:
                original_lr = config["learning_rate"] 
                config["learning_rate"] = original_lr * 0.1  # Use a smaller learning rate for continuation
                print(f"Adjusting learning rate for continuation: {original_lr} -> {config['learning_rate']}")
                model.config["learning_rate"] = config["learning_rate"]
                
            # Make sure the model knows we're continuing from a specific step
            model.train_step_count = args.max_steps
            
            # Create continuation logs
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            checkpoint_logs_path = os.path.join(args.output_dir, "logs", "checkpoint_events.txt")
            with open(checkpoint_logs_path, 'a') as f:
                f.write(f"{timestamp} | CONTINUATION STARTED | {args.max_steps} | Loading from {final_checkpoint_path}\n")
            
            # Create a GPU stats logger that appends instead of overwriting
            class ContinuationGPUStatsLogger(GPUStatsLogger):
                def __init__(self, output_dir):
                    # Initialize without overwriting files
                    super(GPUStatsLogger, self).__init__()
                    self.gpu_logs_path = os.path.join(output_dir, "logs", "gpu_metrics.txt")
                    self.loss_logs_path = os.path.join(output_dir, "logs", "training_loss.txt")
                    self.epoch_logs_path = os.path.join(output_dir, "logs", "epoch_completion.txt")
                    self.checkpoint_logs_path = os.path.join(output_dir, "logs", "checkpoint_events.txt")
                    self.samples_path = os.path.join(output_dir, "logs", "generation_samples.txt")
                    self.last_val_loss = float('nan')
                    self.epoch_start_time = time.time()
                    self.current_epoch = 0
            
            # Create new callbacks that won't overwrite the log files
            continuation_callbacks = [
                ModelCheckpoint(
                    dirpath=os.path.join(args.output_dir, "checkpoints"),
                    filename="smollm2-135-continued-{step}",
                    save_top_k=1,
                    monitor="step",
                    mode="max",
                    every_n_train_steps=args.save_every,
                    save_last=True,
                ),
                LearningRateMonitor(logging_interval="step"),
                TQDMProgressBar(refresh_rate=10),
                ContinuationGPUStatsLogger(args.output_dir)
            ]
            
            # Create continuation validation params - validate once at the end
            continuation_val_params = {"check_val_every_n_epoch": 1}  # Just validate once at the end
                
            # Create a custom StopAtStepsCallback to stop exactly at 50 steps
            class StopAtStepsCallback(pl.Callback):
                def __init__(self, stop_at_steps):
                    super().__init__()
                    self.stop_at_steps = stop_at_steps
                    self.starting_step = None
                    
                def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                    if self.starting_step is None:
                        self.starting_step = trainer.global_step
                        
                    # Check if we've done the required number of steps
                    steps_done = trainer.global_step - self.starting_step
                    if steps_done >= self.stop_at_steps:
                        # Force stop the training
                        print(f"Reached continuation target of {self.stop_at_steps} steps. Stopping training.")
                        trainer.should_stop = True
            
            # Add the callback to stop at exactly 50 steps
            continuation_callbacks.append(StopAtStepsCallback(args.continue_steps))
            
            # Create a new trainer for continuation - use max_epochs=1000 to ensure we don't stop early
            # Actual stopping will be handled by our custom StopAtStepsCallback
            continue_trainer = pl.Trainer(
                max_epochs=1000,  # Use a large number, we'll stop with the callback
                accelerator="auto",
                devices="auto",
                precision=precision,
                callbacks=continuation_callbacks,
                logger=logger,
                log_every_n_steps=10,
                gradient_clip_val=1.0,
                enable_progress_bar=True,
                strategy=strategy,
                **continuation_val_params
            )
            
            print(f"Continuing from step {args.max_steps} for exactly {args.continue_steps} more steps")
            
            # Continue training with the new trainer
            continue_trainer.fit(model, data_module)
            
            # Save final model after continuation
            final_continued_path = os.path.join(args.output_dir, "checkpoints", "final_continued_model.ckpt")
            continue_trainer.save_checkpoint(final_continued_path)
            print(f"Saved final continued model to {final_continued_path}")
            
            # Log completion
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(checkpoint_logs_path, 'a') as f:
                f.write(f"{timestamp} | CONTINUATION COMPLETED | {args.max_steps + args.continue_steps} | Saved to {final_continued_path}\n")
            
            # Training continuation complete
            print(f"Training continuation complete! Total steps: {args.max_steps + args.continue_steps}")
                    
        except Exception as e:
            print(f"Error during continued training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolLM2-135M model")
    parser.add_argument("--config", type=str, default="smoll2.yaml", help="Path to config file")
    parser.add_argument("--input_file", type=str, default="input.txt", help="Path to input text file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of training steps")
    parser.add_argument("--continue_steps", type=int, default=50, help="Number of steps to continue training after checkpoint")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--val_check_interval", type=int, default=None, help="Run validation every N steps")
    parser.add_argument("--validate_by_epoch", action="store_true", help="Run validation every epoch instead of every N steps")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training (overrides config)")
    parser.add_argument("--seq_length", type=int, default=None, help="Sequence length for training (overrides config)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    success = main(args)
    if not success:
        print("Training failed. Please check the error messages above.")
        exit(1)