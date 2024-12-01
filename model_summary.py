from torchinfo import summary
from model import *
import torch

if __name__ == '__main__':
    class Config:
        class AdapterH:
            enable = False
            num_end_adapter_layers = 12
            module_type = "MLP1"

        class LoRA:
            enable = False
            esm_num_end_lora = 16
            r = 8
            alpha = 32
            dropout = 0.05

        class FineTuning:
            enable = False
            unfix_last_layer = 2

        adapter_h = AdapterH()
        lora = LoRA()
        fine_tuning = FineTuning()
        tune_ESM_table = False
        encoder_name = "esm2_t33_650M_UR50D"
        projection_head_name = "LayerNorm"
        hidden_dim = 512
        out_dim = 128
        drop_out = 0.1

    configs = Config()

    sequence_length = 1280
    batch_size = 1
    x = torch.randint(0, 20, (batch_size, sequence_length), dtype=torch.long)

    modes = ['default', 'adapter_h', 'lora', 'finetune']
    for mode in modes:
        if mode == 'adapter_h':
            configs.adapter_h.enable = True
            configs.lora.enable = False
            configs.fine_tuning.enable = False
        elif mode == 'lora':
            configs.adapter_h.enable = False
            configs.lora.enable = True
            configs.fine_tuning.enable = False
        elif mode == 'finetune':
            configs.adapter_h.enable = False
            configs.lora.enable = False
            configs.fine_tuning.enable = True
        else:  # default
            configs.adapter_h.enable = False
            configs.lora.enable = False
            configs.fine_tuning.enable = False

        encoder, projection_head = prepare_models(configs, log_path=None)
        print(f"\nEncoder Summary ({mode} mode):")
        try:
            summary(encoder, input_data=x)
        except Exception as e:
            print(f"Error in {mode} mode: {e}")

    print("\nProjection Head Summary:")
    summary(projection_head, input_size=(1, 1280))

"""
# Colab
!pip install torchinfo
!python ./model_summary.py
"""
