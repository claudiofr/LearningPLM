"""Adapter and LoRA utilities for fine-tuning PLMs."""
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

try:
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class AdapterLayer(nn.Module):
    """Simple adapter layer for transformer models."""

    def __init__(self, input_dim: int, adapter_dim: int, dropout: float = 0.1):
        """
        Initialize adapter layer.
        
        Args:
            input_dim: Dimension of input
            adapter_dim: Dimension of adapter bottleneck
            dropout: Dropout probability
        """
        super().__init__()
        
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.GELU()
        
        # Initialize weights
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        residual = x
        x = self.down_proj(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        
        return x + residual


def add_adapters_to_model(
    model: nn.Module,
    adapter_dim: int = 64,
    layers_to_adapt: Optional[List[int]] = None,
) -> nn.Module:
    """
    Add adapter layers to a transformer model.
    
    Args:
        model: Transformer model
        adapter_dim: Dimension of adapter bottleneck
        layers_to_adapt: List of layer indices to add adapters to (None for all)
        
    Returns:
        Model with adapters
    """
    # Identify transformer layers
    if hasattr(model, "encoder"):
        transformer_layers = model.encoder.layer
    elif hasattr(model, "layers"):
        transformer_layers = model.layers
    else:
        raise ValueError("Could not identify transformer layers in model")
    
    # Set default layers to adapt (last 2 layers)
    if layers_to_adapt is None:
        layers_to_adapt = list(range(len(transformer_layers) - 2, len(transformer_layers)))
    
    # Add adapters to specified layers
    for i in layers_to_adapt:
        if i >= len(transformer_layers):
            raise ValueError(f"Layer index {i} out of range (max: {len(transformer_layers) - 1})")
        
        layer = transformer_layers[i]
        
        # Add adapter after attention
        if hasattr(layer, "attention"):
            output_dim = layer.attention.output.dense.out_features
            layer.attention.output.adapter = AdapterLayer(output_dim, adapter_dim)
            
            # Modify forward pass to include adapter
            original_forward = layer.attention.output.forward
            
            def new_forward(self, hidden_states, input_tensor):
                hidden_states = original_forward(self, hidden_states, input_tensor)
                return self.adapter(hidden_states)
            
            layer.attention.output.forward = new_forward.__get__(layer.attention.output)
        
        # Add adapter after intermediate
        if hasattr(layer, "output"):
            output_dim = layer.output.dense.out_features
            layer.output.adapter = AdapterLayer(output_dim, adapter_dim)
            
            # Modify forward pass to include adapter
            original_forward = layer.output.forward
            
            def new_forward(self, hidden_states, input_tensor):
                hidden_states = original_forward(self, hidden_states, input_tensor)
                return self.adapter(hidden_states)
            
            layer.output.forward = new_forward.__get__(layer.output)
    
    # Freeze all parameters except adapters
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    
    return model


def add_lora_to_model(
    model: PreTrainedModel,
    lora_r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    layers_to_adapt: Optional[List[int]] = None,
) -> PreTrainedModel:
    """
    Add LoRA adapters to a transformer model.
    
    Args:
        model: Transformer model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        layers_to_adapt: List of layer indices to add LoRA to (None for last 2)
        
    Returns:
        Model with LoRA adapters
    """
    if not HAS_PEFT:
        raise ImportError("PEFT library is required for LoRA. Install with: pip install peft")
    
    # Identify transformer layers
    if hasattr(model, "encoder"):
        num_layers = len(model.encoder.layer)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    else:
        raise ValueError("Could not identify transformer layers in model")
    
    # Set default layers to adapt (last 2 layers)
    if layers_to_adapt is None:
        layers_to_adapt = list(range(num_layers - 2, num_layers))
    
    # Create target modules list
    target_modules = []
    for i in layers_to_adapt:
        target_modules.append(f"encoder.layer.{i}.attention.self.query")
        target_modules.append(f"encoder.layer.{i}.attention.self.key")
        target_modules.append(f"encoder.layer.{i}.attention.self.value")
        target_modules.append(f"encoder.layer.{i}.attention.output.dense")
        target_modules.append(f"encoder.layer.{i}.intermediate.dense")
        target_modules.append(f"encoder.layer.{i}.output.dense")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model


def load_model_with_adapters(
    model_name: str,
    adapter_type: str = "lora",
    adapter_dim: int = 64,
    lora_r: int = 4,
    layers_to_adapt: Optional[List[int]] = None,
    device: Optional[str] = None,
    quantize: bool = False,
) -> nn.Module:
    """
    Load a model with adapters.
    
    Args:
        model_name: Name of the model to load
        adapter_type: Type of adapter ("lora" or "adapter")
        adapter_dim: Dimension of adapter bottleneck
        lora_r: LoRA rank
        layers_to_adapt: List of layer indices to add adapters to
        device: Device to load the model on
        quantize: Whether to quantize the model
        
    Returns:
        Model with adapters
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    if quantize:
        if not HAS_PEFT:
            raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")
        
        model = AutoModel.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        model = AutoModel.from_pretrained(model_name).to(device)
    
    # Add adapters
    if adapter_type == "lora":
        model = add_lora_to_model(
            model,
            lora_r=lora_r,
            layers_to_adapt=layers_to_adapt,
        )
    elif adapter_type == "adapter":
        model = add_adapters_to_model(
            model,
            adapter_dim=adapter_dim,
            layers_to_adapt=layers_to_adapt,
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    return model