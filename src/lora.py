"""
Custom from-scratch implementation of LoRA (Low-Rank Adaptation) layers
and helper functions to apply it to a BERT model.
"""

import torch
import torch.nn as nn
import logging
import math

class LoRALinear(nn.Module):
    """
    A LoRA (Low-Rank Adaptation) layer that wraps a standard nn.Linear layer.
    
    This layer freezes the original weights (W) and introduces two new,
    trainable low-rank matrices (A and B) to represent the update (Delta W).
    
    Forward pass computes: h = Wx + (alpha/r) * (xBA)
    """
    def __init__(self, original_linear_layer, r, lora_alpha):
        super().__init__()
        
        self.linear = original_linear_layer
        self.in_features = original_linear_layer.in_features
        self.out_features = original_linear_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Freeze the original layer's weights
        self.linear.weight.requires_grad = False
        
        # Create trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        
        # Scaling factor
        self.scaling = self.lora_alpha / self.r
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is initialized to zeros, so Delta W is zero at the start
        
    def forward(self, x):
        """
        Forward pass:
        1. Compute original output: Wx
        2. Compute LoRA update: (alpha/r) * (xBA)
        3. Return sum
        """
        try:
            # Original, frozen path
            original_output = self.linear(x)
            
            # LoRA path
            lora_update = (x @ self.lora_A @ self.lora_B) * self.scaling
            
            return original_output + lora_update
        except Exception as e:
            logging.error(f"Error in LoRALinear forward pass: {e}")
            raise

def apply_lora_to_bert(model, r, lora_alpha, target_modules=["query", "value"]):
    """
    Recursively applies LoRA to specified target modules within a BERT model.
    
    Replaces target nn.Linear layers with LoRALinear layers.
    """
    logging.info(f"Applying LoRA with r={r}, alpha={lora_alpha} to modules: {target_modules}")
    
    for layer in model.bert.encoder.layer:
        if "query" in target_modules:
            try:
                original_query = layer.attention.self.query
                layer.attention.self.query = LoRALinear(original_query, r, lora_alpha)
            except Exception as e:
                logging.warning(f"Could not apply LoRA to query layer: {e}")
                
        if "value" in target_modules:
            try:
                original_value = layer.attention.self.value
                layer.attention.self.value = LoRALinear(original_value, r, lora_alpha)
            except Exception as e:
                logging.warning(f"Could not apply LoRA to value layer: {e}")
    
    logging.info("LoRA application complete.")
    return model

def mark_only_lora_as_trainable(model):
    """
    Freezes all model parameters except for the LoRA matrices (lora_A, lora_B)
    and the final classifier layer.
    """
    logging.info("Freezing base model and unfreezing LoRA/classifier parameters.")
    
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            
    # Unfreeze the classifier head
    try:
        for param in model.classifier.parameters():
            param.requires_grad = True
    except AttributeError as e:
        logging.error(f"Could not find model.classifier to unfreeze: {e}")
        
    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logging.info(
        f"Trainable params: {trainable_params} | "
        f"All params: {all_param} | "
        f"Trainable %: {100 * trainable_params / all_param:.2f}"
    )
    return trainable_params
