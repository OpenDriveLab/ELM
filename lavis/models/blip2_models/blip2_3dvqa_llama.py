import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from positional_encodings.torch_encodings import PositionalEncoding1D
import numpy as np
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig

from .utils import EasyDict
from transformers import StoppingCriteria, StoppingCriteriaList


@registry.register_model("blip2_vqa_llama")
class Blip2VQALlm(Blip2Base):
    """
    BLIP2 Llm model.
    Supported model types:
        - pretrain_llama7b: fintuned image captioning model with blip2_pretrain_llama7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_llm", "pretrain_flanllmxl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama7b": "configs/models/blip2/blip2_pretrain_llama7b.yaml",
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
        llm_model="stanford_alpaca/llama_7B",
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
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
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
        
        # load some hyperparameters
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.img_start_token = "<Image>"
        self.img_end_token = "</Image>"
        self.end_signal = " "
        print("self.max_txt_len: ", self.max_txt_len)

    def _get_text_len(self, text):
        return self.llm_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def preprocess(self, questions, answers):
        text_input = []
        for q, a in zip(questions, answers):
            conversation = ""
            # rstrip() for the extra " "
            conversation += (
                self.begin_signal + self.role[0] + ": " + 
                self.img_start_token + self.img_end_token + self.end_signal
            )
            conversation += (self.begin_signal + self.role[0] + ": " + q + self.end_signal)
            conversation += (self.begin_signal + self.role[1] + ": " + a + self.end_signal)
            conversation += self.begin_signal
            text_input.append(conversation.rstrip())
        
        return text_input
            

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

        img_embeds = self.llm_proj(query_output.last_hidden_state)
        batch_size, img_len, _ = img_embeds.shape

        # follow VideoChat 
        text_input = self.preprocess(samples["questions"], samples["answers"])

        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token 
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.llm_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.llm_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_before_embeds = self.llm_model.model.embed_tokens(p_before_tokens.input_ids)
            p_after_embeds = self.llm_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.begin_signal + self.role[0] + ": "
            sep2 = self.begin_signal + self.role[1] + ": "
            raw_text = p_after.split(sep2)
            for idx in range(1, len(raw_text)):
                raw_text[idx] = sep2 + raw_text[idx]
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # target: "###Human:       ###Assistant: xxxxx. ###"
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            assert cur_len == answer_targets.shape[1], f"The final length is not equal to the original prompt: {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        
        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.llm_tokenizer.pad_token_id
        inputs_embeds = self.llm_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.llm_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_len)]

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        **kwargs,
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

        img_list = []
        image_emb = self.llm_proj(query_output.last_hidden_state)
        img_list.append(image_emb)

        # llama model
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        chat.messages.append([chat.roles[0], "<Image><ImageHere></Image>\n"])
        self.ask(samples["questions"][0], chat)
        llm_message = self.answer(conv=chat, img_list=img_list, max_new_tokens=200)[0]

        return llm_message

    def predict_answers(
        self,
        samples,
        **kwargs,
    ):
        
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

        img_list = []
        image_emb = self.llm_proj(query_output.last_hidden_state)
        img_list.append(image_emb)

        # llama model
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        chat.messages.append([chat.roles[0], "<Image><ImageHere></Image>\n"])
        self.ask(samples["questions"][0], chat)
        llm_message = self.answer(conv=chat, img_list=img_list, max_new_tokens=200)

        return [llm_message]


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

    def answer(self, conv, img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
                repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        device = img_list[0].device
        stop_words_ids = [
            torch.tensor([835]).to(device),
            torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        conv.messages.append([conv.roles[1], None])
        embs = self.get_context_emb(conv, img_list)
        outputs = self.llm_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
        output_text = self.llm_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text
    
    def ask(self, text, conv):
        conv.messages.append([conv.roles[0], text + '\n'])

    def get_context_emb(self, conv, img_list):
        prompt = self.get_prompt(conv)
        # print(prompt)
        if '<VideoHere>' in prompt:
            prompt_segs = prompt.split('<VideoHere>')
        else:
            prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llm_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).input_ids.to(img_list[0].device)
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.llm_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def get_prompt(self, conv):
        ret = conv.system + conv.sep
        for role, message in conv.messages:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
        return ret



class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


