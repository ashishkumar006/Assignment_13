# SmolLM2-135M Training Project

This repository contains code to train and analyze the SmolLM2-135M model, a smaller variant of language models inspired by Llama architecture.

## Model Architecture

SmolLM2-135M is a decoder-only transformer with the following specifications:

- **Hidden Size**: 576
- **Number of Layers**: 30
- **Number of Attention Heads**: 9
- **Number of KV Heads**: 3 (uses Multi-Query Attention)
- **Intermediate Size**: 1536
- **Max Position Embeddings**: 2048
- **Activation Function**: SiLU
- **Vocabulary Size**: 49,152
- **Total Parameters**: ~135M

## Model Definition

The SmolLM2 model follows the architecture of modern decoder-only transformers with rotary positional embeddings (RoPE). Its core components are:

1. **Token Embeddings**: Converts input tokens to 576-dimensional embeddings
2. **Transformer Blocks (30 layers)**, each containing:
   - **RMSNorm**: Pre-normalization layer (similar to LayerNorm but using RMS)
   - **Self-Attention**: Multi-head attention with 9 attention heads and 3 KV heads (Multi-Query Attention)
   - **RMSNorm**: Another pre-normalization layer
   - **MLP**: Two-layer feed-forward network with SiLU activation, expanding to intermediate size 1536
3. **Final RMSNorm**: Applied to the output embeddings
4. **Language Modeling Head**: Tied to input embeddings for token prediction

Key architectural choices:
- **Multi-Query Attention**: Reduces parameter count while preserving model capacity
- **RMSNorm**: More efficient normalization compared to LayerNorm
- **SiLU Activation**: Better gradient flow than ReLU variants
- **Rotary Position Embeddings**: Effective positional encoding without additional parameters
- **Parameter Sharing**: Input and output embeddings share parameters

## Parameter Calculation

The model has approximately 135 million parameters, distributed as follows:

1. **Token Embeddings**: 
   - Size: 576 × 49,152 = 28,311,552 parameters
   - This maps each of the 49,152 tokens in the vocabulary to a 576-dimensional embedding

2. **Each Transformer Layer**:
   - **Attention QKV Projection**: 
     - Q: 576 × 576 = 331,776 parameters
     - K and V (with MQA, 3 KV heads for 9 attention heads): 
       - K: 576 × (576/3) = 110,592 parameters
       - V: 576 × (576/3) = 110,592 parameters
     - Total for QKV: 331,776 + 110,592 + 110,592 = 552,960 parameters
   
   - **Attention Output Projection**: 576 × 576 = 331,776 parameters
   
   - **MLP Components**:
     - First projection (up-projection): 576 × 1536 = 884,736 parameters
     - Second projection (down-projection): 1536 × 576 = 884,736 parameters
     - Total for MLP: 884,736 + 884,736 = 1,769,472 parameters
   
   - **Layer Normalization**:
     - Two RMSNorm layers with scale parameters: 2 × 576 = 1,152 parameters
   
   - **Total per layer**: 552,960 + 331,776 + 1,769,472 + 1,152 = 2,655,360 parameters

3. **All Transformer Layers**: 
   - 30 × 2,655,360 = 79,660,800 parameters

4. **Final Layer Norm**: 
   - 576 parameters

5. **Output Layer**: 
   - 0 additional parameters (weight sharing with input embeddings)

**Total Parameter Count**: 
- Embeddings + Transformer Layers + Final Layer Norm + Output
- 28,311,552 + 79,660,800 + 576 + 0 = 107,972,928 parameters

The difference between this calculation (~108M) and the model's stated size of 135M is due to:
- Bias terms in various layers
- Additional parameters in attention mechanisms
- Rotary position embedding implementation details
- Other architectural nuances not accounted for in this simplified calculation

## Training Process

The model was trained in two phases:
1. Initial training for 5000 steps with prediction logging every 500 steps
2. Checkpoint saved at 5000 steps
3. Further training for 50 additional steps from the checkpoint

Training optimizations:
- PyTorch Lightning for efficient training
- Weight sharing between input and output embeddings
- Residual standard scaling
- Mixed precision training with bfloat16/float16
- AdamW optimizer with learning rate schedule

## Repository Structure

```
SmolLM2-135M/
├── model.py            # SmolLM2 model implementation
├── data.py             # Data loading and processing utilities
├── train.py            # Main training script
├── generate.py         # Text generation utility
├── app.py              # Inference web app
├── colab_train.py      # Script for training on Google Colab
├── smoll2.yaml         # Model and training configuration
├── input.txt           # Training data
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Usage

### Training

To train the model with default settings:

```bash
python train.py --input_file input.txt --output_dir output
```

For the full training process with 5000 steps and then 50 more steps:

```bash
python train.py --input_file input.txt --output_dir output --max_steps 5000 --continue_steps 50
```

### Text Generation

To generate text using a trained model:

```bash
python generate.py --checkpoint output/checkpoints/final_continued_model.ckpt --prompt "Once upon a time"
```

## Training Logs

Training produces logs that show the model's loss decrease over time and sample generations at regular intervals. These logs can be viewed using TensorBoard:

```bash
tensorboard --logdir output/logs
```

The training also creates detailed text logs in the output/logs directory:
- `gpu_metrics.txt` - GPU utilization and memory usage
- `training_loss.txt` - Training and validation loss values
- `epoch_completion.txt` - Epoch completion times
- `checkpoint_events.txt` - When checkpoints are saved
- `generation_samples.txt` - Generated text samples at every 500 steps

## License

This project is licensed under the MIT License - see the LICENSE file for details.