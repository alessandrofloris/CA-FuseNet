import logging
import torch

logger = logging.getLogger(__name__)


def buildOptimizer(model, training_config):
    
    lr = training_config.get("lr", 1e-4) 
    lr_encoder = training_config.get("lr_encoder", lr) 
    freeze_encoder_epochs = training_config.get("freeze_encoder_epochs", 0) 
    weight_decay = training_config.get("weight_decay", 1e-2)

    has_encoder_group = False

    # Just for logging purposes
    encoder_trainable_count = 0
    encoder_frozen_count = 0

    if hasattr(model, "encoder") and freeze_encoder_epochs > 0:

        encoder_param_ids = {id(p) for p in model.encoder.parameters()} 
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad] 
        
        head_params = [
            p for p in model.parameters() if id(p) not in encoder_param_ids and p.requires_grad
        ] 

        encoder_trainable_count = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        encoder_frozen_count = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
        
        logger.info(
            "freeze_encoder_epochs=%d lr_encoder=%s encoder_trainable_params=%d encoder_frozen_params=%d",
            freeze_encoder_epochs,
            lr_encoder,
            encoder_trainable_count,
            encoder_frozen_count,
        )
        
        if len(encoder_params) == 0:  
            logger.info("All encoder parameters frozen by model config, using single param group for head only")
            optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)    
        else: 
            param_groups = [
                {
                    "params": encoder_params,
                    "lr": 0.0 if freeze_encoder_epochs > 0 else lr_encoder,
                },
                {"params": head_params, "lr": lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
            has_encoder_group = True
    
    else:
        logger.info("Using single param group for the optimizer.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer, has_encoder_group