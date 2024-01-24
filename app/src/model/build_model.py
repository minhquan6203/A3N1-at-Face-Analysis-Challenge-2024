from .model_trans_baseline import Multi_Task_Model_CE
from .model_focal import Multi_Task_Model_Focal
from .svm_model import Multi_Task_Model_SVM
from .model_trans_hydra import Multi_Task_Model_Hydra
from .model_vit_res import Multi_Task_Model_VIT_RES
from typing import List, Dict, Optional,Any

def build_model(config:Dict):
    if config['model']['type_model']=='trans':
        return Multi_Task_Model_CE(config)
    if config['model']['type_model']=='trans_focal':
        return Multi_Task_Model_Focal(config)
    if config['model']['type_model']=='svm':
        return Multi_Task_Model_SVM(config)
    if config['model']['type_model']=='trans_hydra':
        return Multi_Task_Model_Hydra(config)
    if config['model']['type_model']=='vit_res':
        return Multi_Task_Model_VIT_RES(config)
