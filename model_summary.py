from torchinfo import summary
from model import *
import torch

if __name__ == '__main__':
    class Config:
        class AdapterH:
            enable = False

        class LoRA:
            enable = False
            esm_num_end_lora = 33
            r = 8
            alpha = 32
            dropout = 0.05

        class FineTuning:
            enable = False
            unfix_last_layer = 4

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

    encoder, projection_head = prepare_models(configs, log_path=None)

    sequence_length = 1280
    batch_size = 1
    x = torch.randint(0, 20, (batch_size, sequence_length), dtype=torch.long)
    print("Encoder Summary:")
    summary(encoder, input_data=x)

    print("\nProjection Head Summary:")
    summary(projection_head, input_size=(1, 1280))
