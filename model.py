import torch.nn as nn
import torch
import torch.nn.functional as F
import esm
import esm_adapterH
from peft import PeftModel, LoraConfig, get_peft_model

import numpy as np  # for lora


def load_and_add_lora_checkpoint(base_model, lora_checkpoint_path):
    """Add a pretrained LoRa checkpoint to a base model"""
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
    return lora_model


class ESM2(nn.Module):  # embedding table is fixed
    def __init__(self, esm2_pretrain, logging,
                 accelerator,
                 configs,
                 residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096, protein_num_projector=2):
        """
        unfix_last_layer: the number of layers that can be fine-tuned
        """
        super(ESM2, self).__init__()
        if configs.model.esm_encoder.adapter_h.enable:
            if accelerator.is_main_process:
                logging.info("use adapter H")
            # num_end_adapter_layers = configs.model.esm_encoder.adapter_h.num_end_adapter_layers
            adapter_args = configs.model.esm_encoder.adapter_h
            esm2_dict = {
                         "esm2_t36_3B_UR50D": esm_adapterH.pretrained.esm2_t36_3B_UR50D(adapter_args),
                         # 36 layers embedding = 2560
                         "esm2_t33_650M_UR50D": esm_adapterH.pretrained.esm2_t33_650M_UR50D(adapter_args),
                         # 33 layers embedding = 1280
                         "esm2_t30_150M_UR50D": esm_adapterH.pretrained.esm2_t30_150M_UR50D(adapter_args),
                         # 30 layers embedding = 640
                         "esm2_t12_35M_UR50D": esm_adapterH.pretrained.esm2_t12_35M_UR50D(adapter_args),
                         # 12 layers embedding = 480
                         "esm2_t6_8M_UR50D": esm_adapterH.pretrained.esm2_t6_8M_UR50D(adapter_args),
                         # 6 layers embedding = 320
                         }
        else:
            esm2_dict = {
                         "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D(),
                         # 36 layers embedding = 2560
                         "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D(),
                         # 33 layers embedding = 1280
                         "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D(),
                         # 30 layers embedding = 640
                         "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D(),
                         # 12 layers embedding = 480
                         "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D(),
                         # 6 layers embedding = 320
                         }

        # if esm2_pretrain_local is None:
        self.esm2, self.alphabet = esm2_dict[esm2_pretrain]  # alphabet.all_toks
        # else:
        #     print("load esm2 model from local dir")
        #     self.esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_pretrain_local)

        self.num_layers = self.esm2.num_layers
        for p in self.esm2.parameters():  # frozen all parameters first
            p.requires_grad = False

        if configs.model.esm_encoder.adapter_h.enable:
            for name, param in self.esm2.named_parameters():
                if "adapter_layer" in name:
                    # print("unfix adapter_layer")
                    param.requires_grad = True

        if configs.model.esm_encoder.lora.enable:
            if accelerator.is_main_process:
                logging.info('use lora for esm v2')
            if configs.model.esm_encoder.lora.resume.enable:
                self.esm2 = load_and_add_lora_checkpoint(self.esm2, configs.model.esm_encoder.lora.resume)
            else:
                lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
                target_modules = []
                if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                    start_layer_idx = np.max([self.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                    for idx in range(start_layer_idx, self.num_layers):
                        for layer_name in lora_targets:
                            target_modules.append(f"layers.{idx}.{layer_name}")
                    
                peft_config = LoraConfig(
                    inference_mode=False,
                    r=configs.model.esm_encoder.lora.r,
                    lora_alpha=configs.model.esm_encoder.lora.alpha,
                    target_modules=target_modules,
                    lora_dropout=configs.model.esm_encoder.lora.dropout,
                    bias="none",
                    # modules_to_save=modules_to_save
                )
                self.peft_model = get_peft_model(self.esm2, peft_config)
        elif configs.model.esm_encoder.fine_tuning.enable:
            if accelerator.is_main_process:
                logging.info('fine-tune esm v2')
            unfix_last_layer = configs.model.esm_encoder.fine_tuning.unfix_last_layer
            fix_layer_num = self.num_layers - unfix_last_layer
            fix_layer_index = 0
            for layer in self.esm2.layers:  # only fine-tune transformer layers,no contact_head and other parameters
                if fix_layer_index < fix_layer_num:
                    fix_layer_index += 1  # keep these layers frozen
                    continue

                for p in layer.parameters():
                    # logging.info('unfix layer')
                    p.requires_grad = True

            if unfix_last_layer != 0:  # if need fine-tune last layer, the emb_layer_norm_after for last representation should updated
                for p in self.esm2.emb_layer_norm_after.parameters():
                    p.requires_grad = True

        if configs.model.esm_encoder.tune_ESM_table:
            if accelerator.is_main_process:
                logging.info("fine-tune esm embedding parameters")
            for p in self.esm2.embed_tokens.parameters():
                p.requires_grad = True
        
        if hasattr(configs.model.esm_encoder, "MLM"):
            if configs.model.esm_encoder.MLM.enable and configs.model.esm_encoder.MLM.mode == "predict":
                for p in self.esm2.lm_head.parameters():
                    p.requires_grad = True

        self.pool_mode = configs.model.esm_encoder.pool_mode

        self.projectors_residue = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=residue_inner_dim,
                                          num_layers=residue_num_projector,
                                          out_dim=residue_out_dim)

        self.projectors_protein = MoBYMLP(in_dim=self.esm2.embed_dim,
                                          inner_dim=protein_inner_dim,
                                          num_layers=protein_num_projector,
                                          out_dim=protein_out_dim)

    def forward(self, x, return_logits=False, return_embedding=False):
        outputs = self.esm2(x, repr_layers=[self.num_layers], return_contacts=False)
        # print("OKOKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        if return_logits:
            prediction_scores = outputs["logits"]
            return prediction_scores
        else:
            residue_feature = outputs['representations'][self.num_layers]
            # residue_dim = residue_feature.shape #[batch,L,D]
            # average pooling but remove padding tokens
            mask = (x != self.alphabet.padding_idx)  # use this in v2 training
            denom = torch.sum(mask, -1, keepdim=True)
            graph_feature_embedding = torch.sum(residue_feature * mask.unsqueeze(-1), dim=1) / denom  # remove padding
            # print("size of graph_feature_embedding is :")
            # print(len(graph_feature_embedding))
            graph_feature = self.projectors_protein(graph_feature_embedding)  # the cls and eos token is included
            mask = ((x != self.alphabet.padding_idx) & (x != self.alphabet.cls_idx) & (
                    x != self.alphabet.eos_idx))  # use this in v2 training
            residue_feature_embedding = residue_feature[mask]
            residue_feature = self.projectors_residue(residue_feature_embedding)
            # residue_feature = self.projectors_residue(residue_feature.view(-1, residue_dim[-1])).view(residue_dim[0],residue_dim[1],-1)
            if return_embedding:
                return graph_feature, residue_feature, graph_feature_embedding, residue_feature_embedding
            else:
                return graph_feature, residue_feature


def prepare_models(logging, configs, accelerator):
    # Use ESM2 for sequence
    model_seq = ESM2(configs.model.esm_encoder.model_name,
                     accelerator=accelerator,
                     residue_inner_dim=configs.model.esm_encoder.residue_inner_dim,
                     residue_out_dim=configs.model.residue_out_dim,
                     protein_out_dim=configs.model.protein_out_dim,
                     residue_num_projector=configs.model.residue_num_projector,
                     protein_num_projector=configs.model.protein_num_projector,
                     configs=configs, logging=logging)
    
    
    if hasattr(configs.train_settings,"train_lm_head_only"):
            if configs.train_settings.train_lm_head_only is True:
                for name, param in model_struct.named_parameters():
                    param.requires_grad = False
                for name, param in model_seq.named_parameters():
                    param.requires_grad = False
                for p in model_seq.esm2.lm_head.parameters():
                        p.requires_grad = True #will open esm2.embed_tokens.weight!
    
    if accelerator.is_main_process:
        print_trainable_parameters(model_seq, logging)
        print_trainable_parameters(model_struct, logging)

    simclr = SimCLR(model_seq, model_struct, configs=configs)
    return simclr  # model_seq, model_struct #, new_model


if __name__ == '__main__':
    print('test')
