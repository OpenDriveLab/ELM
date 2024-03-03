import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from positional_encodings.torch_encodings import PositionalEncoding1D
import numpy as np
from .modeling_flamingo import FlamingoForConditionalGeneration
from transformers import LlamaTokenizer, LlamaConfig


@registry.register_model("blip2_vqa_flamingo")
class Blip2VQAFlam(Blip2Base):
    """
    BLIP2 Llm model.
    Supported model types:
        - pretrain_flamingo7b: fintuned image captioning model with blip2_pretrain_flamingo7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_llm", "pretrain_flanllmxl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flamingo7b": "configs/models/blip2/blip2_pretrain_flamingo7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="stanford_alpaca/flamingo_7B",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        temporal_length=2,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        self.temporal_length = temporal_length

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # self.Qformer, self.query_tokens, self.extra_query_tokens = self.init_Qformer(num_query_token, 1408, temporal_length=temporal_length)
        self.Qformer, self.query_tokens, _ = self.init_Qformer(num_query_token, 1408, temporal_length=temporal_length)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model)
        self.origin_length = len(self.llm_tokenizer)

        llm_config = LlamaConfig.from_pretrained(llm_model)
        self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, config=llm_config)

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.llm_model.get_output_embeddings().requires_grad_(True)
        self.llm_model.get_input_embeddings().requires_grad_(True)

        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

        self.max_txt_len = max_txt_len
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        # self.attention = nn.MultiheadAttention(emmbed_dim=1408,num_heads=4,batch_first=True)

        self.memory_bank = {}


    def forward(self, samples):
        images = samples["images"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images))
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            images.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(images.device)

        text_input = samples["questions"]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.llm_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(images.device)
            output_tokens = self.llm_tokenizer(
                samples["answers"],
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(images.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_llm = []
            batch_inputs_llm = []

            for b, _ in enumerate(samples["answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_llm += [atts_llm[b]]
                batch_inputs_llm += [inputs_llm[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_llm = torch.stack(batch_atts_llm, dim=0)
            batch_inputs_llm = torch.stack(batch_inputs_llm, dim=0)

            encoder_atts = torch.cat([batch_atts_llm, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.llm_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        images = samples["images"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images))

        slot_embeds = self.slot_attention(image_embeds)
        image_embeds = slot_embeds
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            images.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(images.device)

        text_input = samples["questions"]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.llm_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(images.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_llm = []
            batch_inputs_llm = []

            for b, _ in enumerate(samples["answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_llm += [atts_llm[b]]
                batch_inputs_llm += [inputs_llm[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_llm = torch.stack(batch_atts_llm, dim=0)
            batch_inputs_llm = torch.stack(batch_inputs_llm, dim=0)

            encoder_atts = torch.cat([batch_atts_llm, batch_input_tokens_atts], dim=1)

            inputs_embeds = self.llm_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs,
    ):
        
        images = samples["images"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images))

        slot_embeds = self.slot_attention(image_embeds)
        image_embeds = slot_embeds
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            images.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(images.device)

        text_input = samples["questions"]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.llm_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(images.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_llm = []
            batch_inputs_llm = []

            for b, _ in enumerate(samples["answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_llm += [atts_llm[b]]
                batch_inputs_llm += [inputs_llm[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_llm = torch.stack(batch_atts_llm, dim=0)
            batch_inputs_llm = torch.stack(batch_inputs_llm, dim=0)

            encoder_atts = torch.cat([batch_atts_llm, batch_input_tokens_atts], dim=1)

            inputs_embeds = self.llm_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=1,
                length_penalty=-1,
            )
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True) # False

        if self._apply_lemmatizer:
            output_text_new = self._lemmatize(output_text)
            output_text = output_text_new
            # if output_text_new!=output_text:
            #    print("old: %s, new: %s\n"%(output_text, output_text_new))
        # import pdb; pdb.set_trace()
        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model