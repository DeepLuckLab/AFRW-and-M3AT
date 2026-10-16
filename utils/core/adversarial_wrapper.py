import copy
import torch
from torch import nn
import numpy as np
from losses.nll_loss import NLLSurvLoss

class AdversarialWrapper(nn.Module):
    def __init__(self, **model_kwargs):
        super().__init__()


        self.slide_encoder = copy.deepcopy(model_kwargs['slide_encoder'])
        self.attack_model = model_kwargs.get('attack_model', None)


        self.task_name = model_kwargs.get('task_name', None)
        self.modalities = model_kwargs.get('modalities', 'slide')
        self.model_name = model_kwargs.get('model_name', None)
        self.loss = model_kwargs.get('loss', None)
        self.device = model_kwargs.get('device', 'cuda')
        self.return_raw_attention = model_kwargs.get('return_raw_attention', False)
        self.return_intermediate_feats = model_kwargs.get('return_intermediate_feats', False)
        self.attn_mask = model_kwargs.get('attn_mask', None)
        self.attack_clean = model_kwargs.get("clean_data", False)
        self.adv_training_types = model_kwargs.get("adv_training_types", None)


        self.to(self.device)

        if self.loss is not None:
            if isinstance(self.loss, dict):
                for iter_idx, loss in self.loss.items():
                    self.loss[iter_idx].to(self.device)
            else:

                self.loss.to(self.device)

    def _change_model_mode(self):
        self.slide_encoder.eval()
    def _recover_model_mode(self, given_training):
        if given_training:
            self.slide_encoder.train()

    def _prepare_encoder_input_batch(self, batch):

        slide_cleaned = None
        if "slide" in self.modalities:
            sample_slide_collated = batch["modalities"]["slide"]

            assert len(sample_slide_collated["id"]) == 1, "Batch size must be 1."

            if isinstance(sample_slide_collated["features"], list):
                slide_cleaned = []

                for features, coords, attributes in zip(
                        sample_slide_collated["features"],
                        sample_slide_collated["coords"],
                        sample_slide_collated["attributes"]
                ):

                    if self.attack_model is not None and not self.attack_clean:

                        given_training =  self.slide_encoder.training
                        self._change_model_mode()
                        if self.task_name=="OS":
                            adv_features = self.attack_model(self.slide_encoder, features, [batch["modalities"]["labels"][self.task_name],
                                                                        batch["modalities"]["labels"]["extra_attrs"]])
                        else:
                            adv_features = self.attack_model(self.slide_encoder, features, batch["modalities"]["labels"][self.task_name])
                        self._recover_model_mode(given_training)
                    else:
                        adv_features = None

                    slide_cleaned.append({
                        "slide_features": features,
                        "adv_slide_features": adv_features,
                        "label": batch["modalities"]["labels"][self.task_name],
                        "coords": coords,
                        "attributes": {
                            "patch_size_level0": attributes["patch_size_level0"][0]
                        }
                    })
            else:
                raise NotImplementedError("slide mode only supports list of feature bags.")

        if "omics" in self.modalities:
            omics_cleaned = []
            sample_omics_collated = batch["modalities"]["omics"]
            if isinstance(sample_omics_collated["features"], np.ndarray):
                omics_cleaned.append({
                    "omics_features": torch.from_numpy(sample_omics_collated["features"]).float(),
                    "label": batch["modalities"]["labels"][self.task_name]
                })
            else:
                omics_cleaned.append({
                    "omics_features": sample_omics_collated["features"],
                    "label": batch["modalities"]["labels"][self.task_name]
                })

        samples = []

        if "slide" in self.modalities and "omics" in self.modalities:
            for slide_item in slide_cleaned:
                samples.append({
                    **slide_item,
                    "omics_features": omics_cleaned[0]["omics_features"]
                })

        elif "slide" in self.modalities:
            samples = slide_cleaned

        elif "omics" in self.modalities:
            samples = omics_cleaned

        else:
            raise 'Must be one of "slide", "omics", or "all".'

        return samples

    def _summarize(self, data_cat):

        if not data_cat['logits']: return None

        stack_final_logits = torch.stack(data_cat['logits']).mean(dim=0)
        final_results_dict = {
            "logits": stack_final_logits,
            "logits_list": data_cat['results']
        }
        final_log_dict = {
            "slide_logs": data_cat['logs'],
            "extra_loss_list": data_cat['instance_loss'],
            "extra_loss": None if all(x is None for x in data_cat['instance_loss'])
            else torch.stack([x for x in data_cat['instance_loss'] if x is not None]).mean()
            if any(x is not None for x in data_cat['instance_loss']) else None
        }
        return [final_results_dict, final_log_dict]
    def _mian_run(self, slide_encoder_input):

        if isinstance(slide_encoder_input, list):
            if self.adv_training_types == 'at':
                final_logits = []
                final_extr_loss = []
                final_results = []
                final_logs = []
                for batch in slide_encoder_input:

                    if 'slide' in self.modalities and 'omics' in self.modalities:
                        raise NotImplementedError

                    elif 'slide' in self.modalities:
                        input_key = 'slide_features' if self.attack_clean else 'adv_slide_features'
                        label_in = batch['label'].to(self.device) if self.model_name in ['clam'] else None
                        results_dict, log_dict = self.slide_encoder(features=batch[input_key].to(self.device),
                                                                    return_raw_attention=self.return_raw_attention,
                                                                    return_intermediate_feats=self.return_intermediate_feats,
                                                                    label=label_in,
                                                                    loss_fn=None,
                                                                    attn_mask=self.attn_mask)

                    elif 'omics' in self.modalities:
                        raise NotImplementedError

                    else:
                        raise NotImplementedError

                    final_logits.append(results_dict['logits'])
                    final_results.append(results_dict)
                    final_logs.append(log_dict)
                    final_extr_loss.append(log_dict.get('instance_loss', None))

                stack_final_logits = torch.stack(final_logits).mean(dim=0)

                final_results_dict = {
                    "logits": stack_final_logits,
                    "logits_list": final_results
                }

                final_log_dict = {
                    "slide_logs": final_logs,
                    "extra_loss_list": final_extr_loss,
                    "extra_loss": None if all(x is None for x in final_extr_loss)
                    else torch.stack([x for x in final_extr_loss if x is not None]).mean()
                    if any(x is not None for x in final_extr_loss) else None,
                    "extra_adv_loss": None
                }
                return final_results_dict, final_log_dict

            else:
                raise NotImplementedError

        else:
            raise RuntimeError("Input must be a list of samples")

    def forward(self, batch, output='loss'):




        slide_encoder_input = self._prepare_encoder_input_batch(batch)


        final_results_dict, final_log_dict = self._mian_run(slide_encoder_input)

        logits = final_results_dict['logits']
        extr_loss = final_log_dict['extra_loss']
        extr_adv_loss = final_log_dict['extra_adv_loss']

        if output == 'features':
            if self.attack_model is not None and not self.attack_clean:
                slide_features = slide_encoder_input[0]['adv_slide_features']
            else:
                slide_features = slide_encoder_input[0]['slide_features']
            return slide_features.detach()

        if output == 'attention_raw':
            attention_raw = final_log_dict['slide_logs'][0]['attention_raw']
            return attention_raw.detach().squeeze().cpu().numpy()

        if output == 'logits':
            return logits

        if isinstance(self.loss, NLLSurvLoss):
            y_bins = batch["modalities"]['labels'][self.task_name].to(self.device)
            y_bins = y_bins % 4
            y_event = batch["modalities"]['labels']['extra_attrs'][f'{self.task_name}_event'].to(self.device)
            loss = self.loss(logits, y_bins.unsqueeze(0), y_event.unsqueeze(0))
        elif isinstance(self.loss, dict):
            assert batch.get(
                'current_iter') is not None, "Current iter must be provided for weighted loss, but got None from batch. Please check the dataloader."
            loss = self.loss[batch['current_iter']](logits,
                                                    batch["modalities"]['labels'][self.task_name].to(self.device))
        else:
            loss = self.loss(logits, batch["modalities"]['labels'][self.task_name].to(self.device))

        if output == 'loss':
            info = [{}]
            if extr_loss is not None:
                bag_weight = 0.5
                loss = loss * bag_weight + (1 - bag_weight) * extr_loss
            if extr_adv_loss is not None:
                loss = loss + extr_adv_loss
            return loss, info

        raise ValueError(f"Invalid output type {output}")