import os
import torch
import gradio as gr
import yaml
from transformers import AutoTokenizer
from model import SmolLM2LightningModule, SmolLM2LMHeadModel

# Configuration
MODEL_PATH = "smollm2_model.pt"  # Path to your PyTorch model weights
CONFIG_PATH = "smoll2.yaml"  # Path to your config file
DEFAULT_MAX_LENGTH = 200

# Load model configuration
def load_model_config(config_path):
    """Load model configuration from yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model configuration
        model_config = config["model"]["model_config"]
        return model_config
    except Exception as e:
        print(f"Error loading config: {e}")
        # Return default configuration
        return {
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

# Load the model
def load_model():
    # Load config
    config = load_model_config(CONFIG_PATH)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model from PT file
    try:
        # Create a new model with the config
        model = SmolLM2LightningModule(config, tokenizer)
        
        # Load state dict from PT file
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        
        print(f"Loaded model weights from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a new model with the config as fallback
        model = SmolLM2LightningModule(config, tokenizer)
        print("Created new model with default configuration")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

# Generate text function
def generate_text(prompt, max_length=DEFAULT_MAX_LENGTH, temperature=0.8, top_p=0.9, top_k=50):
    """Generate text from the model based on the input prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to the same device as model
    input_ids = input_ids.to(model.device)
    
    # Generate text
    with torch.no_grad():
        try:
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
        except Exception as e:
            return f"Error generating text: {e}"

# Load the model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = load_model()
print("Model and tokenizer loaded successfully!")

# Create Gradio interface
def inference(prompt, max_length, temperature, top_p, top_k):
    return generate_text(
        prompt, 
        max_length=max_length, 
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

# Define the Gradio interface
demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Slider(minimum=50, maximum=500, value=DEFAULT_MAX_LENGTH, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2-135M Text Generation",
    description="Enter a prompt and the model will generate text based on it. Adjust parameters to control generation.",
    examples=[
        ["Once upon a time in a land far away,", 200, 0.8, 0.9, 50],
        ["The advantages of deep learning include", 300, 0.7, 0.9, 40],
        ["Write a step-by-step guide to baking a chocolate cake", 400, 0.9, 0.95, 50],
        ["In the year 2050, technology has advanced to the point where", 250, 1.0, 0.9, 50]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()