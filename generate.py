import os
import argparse
import torch
from transformers import AutoTokenizer
from model import SmolLM2LightningModule
import pytorch_lightning as pl
import yaml


def load_model_config(config_path):
    """Load model configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model configuration
    model_config = config["model"]["model_config"]
    
    return model_config


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9, top_k=50):
    """Generate text from the model."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to the same device as model
    input_ids = input_ids.to(model.device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def main(args):
    # Load model configuration
    if args.config:
        config = load_model_config(args.config)
        print(f"Loaded model config from {args.config}")
    else:
        # Default configuration
        config = {
            "hidden_size": 576,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "num_hidden_layers": 30,
            "intermediate_size": 1536,
            "max_position_embeddings": 2048,
            "vocab_size": 49152,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "initializer_range": 0.042,
            "tie_word_embeddings": True,
            "use_cache": True,
            "hidden_act": "silu",
        }
        print("Using default model config")
    
    # Load tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path}")
    else:
        # Use standard HF Llama tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        print("Using default Llama tokenizer")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    if args.checkpoint:
        model = SmolLM2LightningModule.load_from_checkpoint(args.checkpoint)
        print(f"Loaded model from checkpoint {args.checkpoint}")
    else:
        # Create a new model
        model = SmolLM2LightningModule(config, tokenizer)
        print("Created new model with default configuration")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Process prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    else:
        prompt = "Once upon a time"
    
    print("\nPrompt:")
    print(prompt)
    
    # Generate text
    print("\nGenerating text...")
    generated_text = generate_text(
        model,
        tokenizer,
        prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    
    print("\nGenerated Text:")
    print(generated_text)
    
    # Save generated text to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nSaved generated text to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with SmolLM2-135M model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for text generation")
    parser.add_argument("--prompt_file", type=str, default=None, help="File containing prompt for text generation")
    parser.add_argument("--output_file", type=str, default=None, help="File to save generated text")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    
    main(args) 