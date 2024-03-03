import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D
import numpy as np
from lavis.datasets.data_utils import convert_length_to_mask

import re
import torch.nn.functional as F
import copy, random


@registry.register_model("blip2_vqa_t5_ego4d")
class Blip2VQAT5Ego(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
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
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        temporal_length=3,
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

        self.Qformer, self.query_tokens, _ = self.init_Qformer(num_query_token, 1408, temporal_length=temporal_length)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        self.feat_adapter = nn.Linear(1536, 1408, bias=False)        
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        self.origin_length = len(self.t5_tokenizer)

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        # print(len(self.t5_tokenizer.vocab))
        # print(f'{self.t5_tokenizer.all_special_tokens=}')
        # input()

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        # self.attention = nn.MultiheadAttention(emmbed_dim=1408,num_heads=4,batch_first=True)

        self.memory_bank = {}
        del self.visual_encoder, self.ln_vision

    def find_adj(self, B, adj_id, scene_id, image_embeds):
        
        batch_adj_img = []
        for i in range(B):
            adj_img_embeds = []
            for single_adj_id in adj_id[i]:
                # print(single_adj_id, self.memory_bank.keys())
                # single_adj_id = scene_id[i]
                if single_adj_id in self.memory_bank:
                    adj_img_embeds.append(self.memory_bank[single_adj_id])
            if len(adj_img_embeds) == len(adj_id[i]):
                adj_imgs = torch.stack(adj_img_embeds, dim=0)
                # print(adj_imgs.shape)
                batch_adj_img.append(adj_imgs)
        

            self.memory_bank[scene_id[i]] = image_embeds[i].cpu()
        # print(self.memory_bank.keys())
        # print(len(batch_adj_img))
        if len(batch_adj_img) == B:
            return torch.stack(batch_adj_img, dim=0) # B T N C
        else:
            return None

    def location_token_init(self):
        location_tokens = []
        for i in range(510):  # pc -> x,y,z tokens
            location_tokens.append("<loc%d>" % i)  # bbox_min: 500; bbox_max: 550; (xmin, ymin, zmin) (xmax, ymax, zmax)
        for i in range(64):  # pc -> x,y,z tokens
            location_tokens.append("<rad%d>" % i)  # bbox_min: 500; bbox_max: 550; (xmin, ymin, zmin) (xmax, ymax, zmax)
        for i in range(64):  # pc -> x,y,z tokens
            location_tokens.append("<dim%d>" % i)  # bbox_min: 500; bbox_max: 550; (xmin, ymin, zmin) (xmax, ymax, zmax)
        for i in range(64):  # pc -> x,y,z tokens
            location_tokens.append("<velo%d>" % i)  # bbox_min: 500; bbox_max: 550; (xmin, ymin, zmin) (xmax, ymax, zmax)
        return location_tokens
    
    def add_pos_embed(self):

        pos_embed = self.get_2d_sincos_pos_embed(self.t5_model.config.hidden_size,510,4)
        # print(pos_embed.shape)
        input_embeddings = self.t5_model.get_input_embeddings()
        # print(input_embeddings)
        # print(input_embeddings.weight.shape)
        # print(self.t5_model.embed_tokens.word_embeddings.weight)
        # print(self.t5_model.embeddings.word_embeddings.weight[-1, :])
        index = self.origin_length-1
        # print(input_embeddings.weight[32100:32105])
        with torch.no_grad():
            for i in range(510):
                index += 1
                input_embeddings.weight[index, :] += pos_embed[i,0, :]
            for i in range(64):
                index += 1
                input_embeddings.weight[index, :] += pos_embed[i,1, :]
            for i in range(64):
                index += 1
                input_embeddings.weight[index, :] += pos_embed[i,2, :]
            for i in range(64):
                index += 1
                input_embeddings.weight[index, :] += pos_embed[i,3, :]
        # print(input_embeddings.weight[32100:32105])
        # input()
        self.t5_model.set_input_embeddings(input_embeddings)


    def get_2d_sincos_pos_embed(self, embed_dim, size_h, size_w):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(size_h, dtype=np.float32)
        grid_w = np.arange(size_w, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        # grid = grid.reshape([2, 1, size_h, grid_w])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid).reshape([size_h, size_w, embed_dim])

        return torch.from_numpy(pos_embed)


    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        # assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb
    
    
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb
            
    
    def forward(self, samples):
        device = samples["vfeats"].device
        records, vfeats, vfeat_lens = samples["records"], samples["vfeats"], samples["vfeat_lens"]
        word_ids, char_ids, s_labels = samples["word_ids"], samples["char_ids"], samples["s_labels"]
        e_labels, h_labels, question = samples["e_labels"], samples["h_labels"], samples["question"]
        word_ids = {key: val.to(device) for key, val in word_ids.items()}
        
        # generate mask
        video_mask = convert_length_to_mask(vfeat_lens).to(device)

        B = s_labels.shape[0]
        device = vfeats.device
        with self.maybe_autocast():
            image_embeds = vfeats # [2, 128, 1536]
            image_embeds = self.feat_adapter(image_embeds) #[2, 128, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = []
            for i in range(query_tokens.shape[1]):
                query_output.append(
                    self.Qformer.bert(
                        query_embeds=query_tokens[:, i:i+1],
                        encoder_hidden_states=image_embeds[:, 4*i:4*i+4],
                        encoder_attention_mask=image_atts[:, 4*i:4*i+4],
                        return_dict=True,).last_hidden_state
                    )
            query_output = torch.cat(query_output, dim=1)
            t5_query = query_output#.last_hidden_state

        inputs_t5 = self.t5_proj(t5_query)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        gt_output_text = samples["gt_output_text"]
        
        # question + prompt
        text_input = samples["question"]
        for i in range(B):
            text_input[i] += " The input feature include 32 frames. please tell me which frame is the start frame of the action."

        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer( # 8 x 17
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(device)
            output_tokens = self.t5_tokenizer( # 8 x 17
                gt_output_text,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(device)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, _ in enumerate(range(B)):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_t5 += [atts_t5[b]]
                batch_inputs_t5 += [inputs_t5[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            
            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model(
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
        device = samples["vfeats"].device
        records, vfeats, vfeat_lens = samples["records"], samples["vfeats"], samples["vfeat_lens"]
        word_ids, char_ids, s_labels = samples["word_ids"], samples["char_ids"], samples["s_labels"]
        e_labels, h_labels, question = samples["e_labels"], samples["h_labels"], samples["question"]
        word_ids = {key: val.to(device) for key, val in word_ids.items()}
        
        # generate mask
        video_mask = convert_length_to_mask(vfeat_lens).to(device)

        B = s_labels.shape[0]
        device = vfeats.device
        with self.maybe_autocast():
            image_embeds = vfeats # [2, 128, 1536]
            image_embeds = self.feat_adapter(image_embeds) #[2, 128, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = []
            for i in range(query_tokens.shape[1]):
                query_output.append(
                    self.Qformer.bert(
                        query_embeds=query_tokens[:, i:i+1],
                        encoder_hidden_states=image_embeds[:, 4*i:4*i+4],
                        encoder_attention_mask=image_atts[:, 4*i:4*i+4],
                        return_dict=True,).last_hidden_state
                    )
            query_output = torch.cat(query_output, dim=1)
            t5_query = query_output#.last_hidden_state

        inputs_t5 = self.t5_proj(t5_query)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        gt_output_text = samples["gt_output_text"]
        
        # question + prompt
        text_input = samples["question"]
        for i in range(B):
            text_input[i] += " The input feature include 32 frames. please tell me which frame is the start frame of the action."
        
        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer( # 8 x 17
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(device)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, _ in enumerate(range(B)):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_t5 += [atts_t5[b]]
                batch_inputs_t5 += [inputs_t5[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
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
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
        
        device = samples["vfeats"].device
        records, vfeats, vfeat_lens = samples["records"], samples["vfeats"], samples["vfeat_lens"]
        word_ids, char_ids, s_labels = samples["word_ids"], samples["char_ids"], samples["s_labels"]
        e_labels, h_labels, question = samples["e_labels"], samples["h_labels"], samples["question"]
        word_ids = {key: val.to(device) for key, val in word_ids.items()}
        
        # generate mask
        video_mask = convert_length_to_mask(vfeat_lens).to(device)

        B = s_labels.shape[0]
        device = vfeats.device
        with self.maybe_autocast():
            image_embeds = vfeats # [2, 128, 1536]
            image_embeds = self.feat_adapter(image_embeds) #[2, 128, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = []
            for i in range(query_tokens.shape[1]):
                query_output.append(
                    self.Qformer.bert(
                        query_embeds=query_tokens[:, i:i+1],
                        encoder_hidden_states=image_embeds[:, 4*i:4*i+4],
                        encoder_attention_mask=image_atts[:, 4*i:4*i+4],
                        return_dict=True,).last_hidden_state
                    )
            query_output = torch.cat(query_output, dim=1)
            t5_query = query_output#.last_hidden_state

        inputs_t5 = self.t5_proj(t5_query)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        gt_output_text = samples["gt_output_text"]
        
        # question + prompt
        text_input = samples["question"]
        for i in range(B):
            text_input[i] += " The input feature include 32 frames. please tell me which frame is the start frame of the action."
        
        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer( # 8 x 17
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(device)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, _ in enumerate(range(B)):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_t5 += [atts_t5[b]]
                batch_inputs_t5 += [inputs_t5[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)
            
            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=1,
                length_penalty=-1,
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

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
        t5_model = cfg.get("t5_model")

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
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
