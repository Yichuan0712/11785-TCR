import torch.nn as nn
import torch
import torch.nn.functional as F
import esm
import esm_adapterH
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np  # for lora


class ESM2(nn.Module):  # embedding table is fixed
    def __init__(self, configs):
        super(ESM2, self).__init__()
        esm2_dict = {}
        if configs.adapter_h.enable:
            adapter_args = configs.adapter_h
            if configs.encoder_name == "esm2_t36_3B_UR50D":
                esm2_dict["esm2_t36_3B_UR50D"] = esm_adapterH.pretrained.esm2_t36_3B_UR50D(adapter_args)
            elif configs.encoder_name == "esm2_t33_650M_UR50D":
                esm2_dict["esm2_t33_650M_UR50D"] = esm_adapterH.pretrained.esm2_t33_650M_UR50D(adapter_args)
            elif configs.encoder_name == "esm2_t30_150M_UR50D":
                esm2_dict["esm2_t30_150M_UR50D"] = esm_adapterH.pretrained.esm2_t30_150M_UR50D(adapter_args)
            elif configs.encoder_name == "esm2_t12_35M_UR50D":
                esm2_dict["esm2_t12_35M_UR50D"] = esm_adapterH.pretrained.esm2_t12_35M_UR50D(adapter_args)
            elif configs.encoder_name == "esm2_t6_8M_UR50D":
                esm2_dict["esm2_t6_8M_UR50D"] = esm_adapterH.pretrained.esm2_t6_8M_UR50D(adapter_args)
            else:
                raise ValueError(f"Unknown encoder name: {configs.encoder_name}")
        else:
            if configs.encoder_name == "esm2_t36_3B_UR50D":
                esm2_dict["esm2_t36_3B_UR50D"] = esm.pretrained.esm2_t36_3B_UR50D()
            elif configs.encoder_name == "esm2_t33_650M_UR50D":
                esm2_dict["esm2_t33_650M_UR50D"] = esm.pretrained.esm2_t33_650M_UR50D()
            elif configs.encoder_name == "esm2_t30_150M_UR50D":
                esm2_dict["esm2_t30_150M_UR50D"] = esm.pretrained.esm2_t30_150M_UR50D()
            elif configs.encoder_name == "esm2_t12_35M_UR50D":
                esm2_dict["esm2_t12_35M_UR50D"] = esm.pretrained.esm2_t12_35M_UR50D()
            elif configs.encoder_name == "esm2_t6_8M_UR50D":
                esm2_dict["esm2_t6_8M_UR50D"] = esm.pretrained.esm2_t6_8M_UR50D()
            else:
                raise ValueError(f"Unknown encoder name: {configs.encoder_name}")

        self.esm2, self.alphabet = esm2_dict[configs.encoder_name]

        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters():
            p.requires_grad = False

        if configs.adapter_h.enable:
            for name, param in self.esm2.named_parameters():
                if "adapter_layer" in name:
                    param.requires_grad = True

        if configs.lora.enable:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
            target_modules = []
            if configs.lora.esm_num_end_lora > 0:
                start_layer_idx = np.max([self.num_layers - configs.lora.esm_num_end_lora, 0])
                for idx in range(start_layer_idx, self.num_layers):
                    for layer_name in lora_targets:
                        target_modules.append(f"layers.{idx}.{layer_name}")

            peft_config = LoraConfig(
                inference_mode=False,
                r=configs.lora.r,
                lora_alpha=configs.lora.alpha,
                target_modules=target_modules,
                lora_dropout=configs.lora.dropout,
                bias="none",
            )
            self.peft_model = get_peft_model(self.esm2, peft_config)

        elif configs.fine_tuning.enable:
            unfix_last_layer = configs.fine_tuning.unfix_last_layer  # unfix_last_layer: the number of layers that can be fine-tuned
            fix_layer_num = self.num_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.esm2.layers:  # only fine-tune transformer layers, no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    p.requires_grad = True

            if unfix_last_layer != 0:  # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
                for p in self.esm2.emb_layer_norm_after.parameters():
                    p.requires_grad = True

        if configs.tune_ESM_table:
            for p in self.esm2.embed_tokens.parameters():
                p.requires_grad = True

    def forward(self, x):
        outputs = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)

        residue_feature = outputs['representations'][self.num_layers]

        return residue_feature


def prepare_models(configs, log_path):
    # Use ESM2 for sequence
    model_seq = ESM2(configs)

    return None


if __name__ == '__main__':
    print('test')
