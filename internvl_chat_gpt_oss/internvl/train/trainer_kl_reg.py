# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Custom Trainer with KL Divergence Regularization
# --------------------------------------------------------

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)


class InternVLTrainerWithKLReg(Trainer):
    """
    Custom Trainer that adds KL divergence regularization to maintain similarity
    with a reference (pretrained) model during finetuning.
    
    This helps prevent catastrophic forgetting and performance degradation on
    existing benchmarks after finetuning.
    """
    
    def __init__(
        self,
        *args,
        ref_model: Optional[nn.Module] = None,
        kl_coeff: float = 0.1,
        **kwargs
    ):
        """
        Args:
            ref_model: Reference (pretrained) model for KL divergence computation.
                      If None, KL regularization is disabled.
            kl_coeff: Coefficient for KL divergence term in the loss (beta in DPO).
                     Higher values = stronger regularization = stay closer to reference model.
        """
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        
        if self.ref_model is not None:
            # Ensure reference model is in eval mode and frozen
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            
            # Important: Detach from DDP/DeepSpeed - don't let them wrap this model
            # The ref_model should NOT participate in any collective operations
            if hasattr(self.ref_model, 'module'):
                logger.warning('Reference model appears to be wrapped in DDP/DataParallel - unwrapping')
                self.ref_model = self.ref_model.module
            
            logger.info(f'KL Regularization enabled with coefficient: {kl_coeff}')
        else:
            logger.info('KL Regularization disabled (no reference model provided)')
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to add KL divergence regularization.
        
        Total Loss = Standard Loss + kl_coeff * KL(policy || reference)
        """
        # Standard forward pass and loss computation
        outputs = model(**inputs)
        
        # Standard loss (e.g., cross-entropy for language modeling)
        if "labels" in inputs:
            loss = outputs.loss
        else:
            # If no labels, we might be in eval mode
            if return_outputs:
                return (outputs.loss, outputs) if hasattr(outputs, 'loss') else (None, outputs)
            return outputs.loss if hasattr(outputs, 'loss') else None
        
        # Add KL divergence regularization if reference model is available
        if self.ref_model is not None and self.kl_coeff > 0:
            kl_loss = self._compute_kl_divergence(model, inputs, outputs)
            total_loss = loss + self.kl_coeff * kl_loss
            
            # Log the losses for monitoring (only on rank 0 to avoid sync issues)
            if self.state.global_step % self.args.logging_steps == 0 and self.args.local_rank in [0, -1]:
                # Detach before calling .item() to avoid gradient graph issues
                logger.info(
                    f'Step {self.state.global_step}: '
                    f'Standard Loss: {loss.detach().item():.4f}, '
                    f'KL Loss: {kl_loss.detach().item():.4f}, '
                    f'Total Loss: {total_loss.detach().item():.4f}'
                )
        else:
            total_loss = loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_kl_divergence(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        policy_outputs,
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy model and reference model.
        
        KL(policy || ref) = sum(policy_probs * (log(policy_probs) - log(ref_probs)))
        
        Args:
            model: The policy (finetuning) model
            inputs: Input batch
            policy_outputs: Outputs from the policy model
            
        Returns:
            KL divergence scalar tensor
        """
        # CRITICAL: Ensure no gradient tracking for reference model
        # This prevents it from participating in DeepSpeed's backward pass
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # Convert inputs to float32 temporarily for numerical stability
            # and to avoid DeepSpeed's mixed precision hooks
            ref_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    ref_inputs[k] = v.float()
                else:
                    ref_inputs[k] = v
            
            # Get reference model predictions
            ref_outputs = self.ref_model(**ref_inputs)
        
        # Get logits from both models
        policy_logits = policy_outputs.logits  # [batch, seq_len, vocab_size]
        ref_logits = ref_outputs.logits.to(policy_logits.dtype).to(policy_logits.device)
        
        # Get labels to mask padding tokens
        labels = inputs.get("labels", None)
        
        if labels is not None:
            # Create mask for valid (non-padding) tokens
            # Typically, labels have -100 for tokens we don't want to compute loss on
            mask = (labels != -100).float()
            
            # Shift logits and labels for next-token prediction alignment
            # policy_logits: [batch, seq_len, vocab]
            # We need to align with labels which are shifted by 1
            shift_logits_policy = policy_logits[..., :-1, :].contiguous()
            shift_logits_ref = ref_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = mask[..., 1:].contiguous()
            
            # Compute log probabilities
            log_probs_policy = F.log_softmax(shift_logits_policy, dim=-1)
            log_probs_ref = F.log_softmax(shift_logits_ref, dim=-1)
            
            # Replace -100 with 0 for gather operation (will be masked out anyway)
            valid_labels = shift_labels.clone()
            valid_labels[valid_labels == -100] = 0
            
            # Get probabilities for the actual tokens (using labels as indices)
            # Shape: [batch, seq_len]
            log_probs_policy_selected = torch.gather(
                log_probs_policy,
                dim=-1,
                index=valid_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            log_probs_ref_selected = torch.gather(
                log_probs_ref,
                dim=-1,
                index=valid_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # KL divergence: KL(policy || ref) for selected tokens
            # In practice, we use: log(policy) - log(ref) for the actual tokens
            # This is an approximation focusing on the target tokens
            kl_div = (log_probs_policy_selected - log_probs_ref_selected) * shift_mask
            
            # Average over valid tokens
            kl_loss = kl_div.sum() / (shift_mask.sum() + 1e-8)
        else:
            # Fallback: compute full KL divergence over vocabulary
            # KL(P||Q) = sum_i P(i) * (log P(i) - log Q(i))
            log_probs_policy = F.log_softmax(policy_logits, dim=-1)
            log_probs_ref = F.log_softmax(ref_logits, dim=-1)
            probs_policy = F.softmax(policy_logits, dim=-1)
            
            kl_div = probs_policy * (log_probs_policy - log_probs_ref)
            kl_loss = kl_div.sum(dim=-1).mean()
        
        return kl_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to handle reference model during evaluation.
        We don't need KL loss during evaluation.
        """
        # Temporarily disable KL computation during evaluation
        original_kl_coeff = self.kl_coeff
        self.kl_coeff = 0.0
        
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        # Restore KL coefficient
        self.kl_coeff = original_kl_coeff
        
        return result
