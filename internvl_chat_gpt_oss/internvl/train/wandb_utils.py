"""
Weights & Biases integration utilities for InternVL training.
"""

import os
import logging
from dataclasses import asdict
from typing import Optional, List

import numpy as np
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class WandbCallback(TrainerCallback):
    """TrainerCallback that forwards training logs to Weights & Biases."""

    def __init__(self, wandb_module):
        self.wandb = wandb_module

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Forward logs to wandb on every logging step."""
        if logs is None:
            return
        try:
            # Convert numpy types to Python types for JSON serialization
            safe_logs = {}
            for k, v in logs.items():
                if isinstance(v, (int, float, np.number)):
                    safe_logs[k] = float(v)
                elif isinstance(v, str):
                    safe_logs[k] = v
                # Skip non-serializable items
            
            self.wandb.log(safe_logs, step=state.global_step)
        except Exception as e:
            logger.warning(f'Failed to send logs to wandb: {e}')


def init_wandb(training_args, model_args=None, data_args=None):
    """
    Initialize Weights & Biases if enabled.
    
    Args:
        training_args: TrainingArguments instance
        model_args: Optional ModelArguments instance
        data_args: Optional DataTrainingArguments instance
    
    Returns:
        tuple: (wandb_enabled: bool, wandb_callbacks: List[TrainerCallback])
    """
    # Check if wandb should be enabled
    use_wandb = (
        ('wandb' in getattr(training_args, 'report_to', [])) or 
        os.environ.get('USE_WANDB', '0') == '1'
    )
    
    if not use_wandb:
        logger.info('W&B integration is disabled')
        return False, []
    
    try:
        import wandb
        
        # Build config from dataclasses
        config = {}
        if model_args is not None:
            try:
                config.update(asdict(model_args))
            except Exception as e:
                logger.warning(f'Failed to convert model_args to dict: {e}')
        
        if data_args is not None:
            try:
                config.update(asdict(data_args))
            except Exception as e:
                logger.warning(f'Failed to convert data_args to dict: {e}')
        
        if training_args is not None:
            try:
                config.update(asdict(training_args))
            except Exception as e:
                logger.warning(f'Failed to convert training_args to dict: {e}')
        
        # Get project name and run name
        project = os.environ.get('WANDB_PROJECT', 'InternVL')
        run_name = training_args.run_name or os.path.basename(training_args.output_dir)
        
        # Initialize wandb
        wandb.init(
            project=project,
            name=run_name,
            config=config,
            reinit=True,
            resume='allow',  # Allow resuming from checkpoints
        )
        
        logger.info(f'✓ W&B initialized successfully')
        logger.info(f'  - Project: {project}')
        logger.info(f'  - Run name: {run_name}')
        logger.info(f'  - URL: {wandb.run.get_url()}')
        
        # Create callback
        callback = WandbCallback(wandb)
        
        return True, [callback]
        
    except ImportError:
        logger.warning('W&B is enabled but wandb package is not installed. Install with: pip install wandb')
        return False, []
    except Exception as e:
        logger.warning(f'W&B initialization failed: {e}')
        return False, []


def finish_wandb():
    """Finish the wandb run gracefully."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            logger.info('W&B run finished')
    except Exception as e:
        logger.warning(f'Failed to finish wandb run: {e}')
