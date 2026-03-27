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

def buildMultiModalOptimizer(model, training_config):
    """
    Versione evoluta che integra la distinzione tra video/pose encoder 
    con la logica di gestione del freeze per epoche.
    """
    # 1. Estrazione parametri dalla configurazione
    lr = training_config.get("lr", 1e-4)
    lr_encoder = training_config.get("lr_encoder", lr)
    lr_gate = training_config.get("lr_gate", lr * 0.01)
    freeze_encoder_epochs = training_config.get("freeze_encoder_epochs", 0)
    weight_decay = training_config.get("weight_decay", 1e-2)

    # 2. Identificazione parametri Encoder (Video + Pose)
    encoder_params = []
    encoder_param_ids = set()
    
    # Iteriamo su entrambi i componenti come nel Metodo 1
    for submodule in [getattr(model, "video_encoder", None), getattr(model, "pose_encoder", None)]:
        if submodule is not None:
            for param in submodule.parameters():
                if param.requires_grad:
                    p_id = id(param)
                    if p_id not in encoder_param_ids:
                        encoder_params.append(param)
                        encoder_param_ids.add(p_id)

    # Gate MLP params
    gate_params = []
    gate_param_ids = set()
    gating_module = getattr(model, "gating", None)
    if gating_module is not None:
        gate_mlp = getattr(gating_module, "gate_mlp", None)
        if gate_mlp is not None:
            for param in gate_mlp.parameters():
                if param.requires_grad:
                    p_id = id(param)
                    if p_id not in encoder_param_ids:
                        gate_params.append(param)
                        gate_param_ids.add(p_id)

    # 3. Identificazione parametri Fusion/Head (Tutto il resto)
    excluded_ids = encoder_param_ids | gate_param_ids
    fusion_params = [
        p for p in model.parameters()
        if id(p) not in excluded_ids and p.requires_grad
    ]

    # 4. Logging e Statistiche
    num_encoder_vars = sum(p.numel() for p in encoder_params)
    num_fusion_vars = sum(p.numel() for p in fusion_params)
    num_gate_vars = sum(p.numel() for p in gate_params)
    
    logger.info(
            f"Optimizer Setup: Encoder params: {num_encoder_vars} | "
            f"Fusion params: {num_fusion_vars} | Gate params: {num_gate_vars}"
        )
    
    # 5. Creazione Param Groups
    # Se freeze_encoder_epochs > 0, partiamo con LR 0.0 per gli encoder
    current_lr_encoder = 0.0 if freeze_encoder_epochs > 0 else lr_encoder

    param_groups = [
        {"params": encoder_params, "lr": current_lr_encoder, "name": "encoder"},
        {"params": fusion_params, "lr": lr, "name": "fusion"},
        {"params": gate_params, "lr": lr_gate, "name": "gate"},
    ]

    # 6. Inizializzazione Ottimizzatore
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    has_encoder_group = len(encoder_params) > 0
    return optimizer, has_encoder_group