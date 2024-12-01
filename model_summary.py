from torchinfo import summary
from model import *

if __name__ == '__main__':
    class Config:
        class AdapterH:
            enable = False

        class LoRA:
            enable = False
            esm_num_end_lora = 0
            r = 8
            alpha = 16
            dropout = 0.1

        class FineTuning:
            enable = False
            unfix_last_layer = 0

        adapter_h = AdapterH()
        lora = LoRA()
        fine_tuning = FineTuning()
        tune_ESM_table = False
        encoder_name = "esm2_t33_650M_UR50D"
        projection_head_name = "LayerNorm"
        hidden_dim = 256
        out_dim = 128
        drop_out = 0.1

    configs = Config()

    encoder, projection_head = prepare_models(configs, log_path=None)

    print("Encoder Summary:")
    summary(encoder, input_size=(1, 1280))

    print("\nProjection Head Summary:")
    summary(projection_head, input_size=(1, 1280))
