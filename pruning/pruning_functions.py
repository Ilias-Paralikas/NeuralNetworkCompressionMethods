import torch 
import torch.nn as nn
from nni.compression.pruning import L1NormPruner,L2NormPruner,FPGMPruner
from nni.compression.speedup import ModelSpeedup




def prune_model(model, 
                sparse_ratio, 
                input_shape, 
                device='cpu',
                op_types=['Linear','Conv2d','Conv3d'], 
                exclude_op_names=None,
                prunner_choice =  None):
  config_list = [{}]
  config_list [0]['sparse_ratio'] = sparse_ratio
  for name, _ in model.named_modules():
    pass
  config_list[0]['op_types'] = op_types
  if not exclude_op_names:
     config_list[0]['exclude_op_names'] = [name]
  else:
    config_list[0]['exclude_op_names'] = exclude_op_names
  
  if not prunner_choice:
    prunner_choice = L1NormPruner
    
  pruner = prunner_choice(model, config_list)
  _, masks = pruner.compress()
  pruner.unwrap_model()
  ModelSpeedup(model, torch.rand(input_shape).to(device), masks).speedup_model()
  print(model)

