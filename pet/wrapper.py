"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""
import json
import os
from typing import List, _TypedDict, Dict

import jsonpickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import  DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    LlamaModel, 
    LlamaConfig,
    PreTrainedTokenizerFast

)

import pet.log as log
from pet.preprocessor import Preprocessor
from pet.utils import InputFeatures, DictDataset, InputAugExample

import random

from peft import (
    get_peft_model, TaskType, LoraConfig, PeftModel)

logger = log.get_logger("root")

CONFIG_NAME = "wrapper_config.json"

MODEL_CLASSES = {
    "scibert": {
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "backbone": AutoModelForMaskedLM,
    },
    "llama":{
        "config": LlamaConfig,
        "tokenizer": PreTrainedTokenizerFast,
        "backbone": LlamaModel,
    },
}


class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(
            self,
            model_type: str,
            model_name_or_path: str,
            dataset: str,
            max_seq_length: int,
            label_list: List[str],
            l2=0,
            l3=0,
            t2=None,
            t3=None,
            hidden_dim=1024,
            emb_dim=256,
            lora_r=8,
            lora_a=32,
            encoder_dropout=0.05,
            encoder_output_dim = 512
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.loggers_initialized = False
        self.dataset = dataset
        
        self.l2 = l2
        self.t2 = t2
        self.l3 = l3
        self.t3 = t3
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.lora_r = lora_r
        self.lora_a = lora_a
        self.encoder_dropout = encoder_dropout
        self.encoder_output_dim = encoder_output_dim


class Citss:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config
        
        config_class = MODEL_CLASSES[self.config.model_type]["config"]
        tokenizer_class = MODEL_CLASSES[self.config.model_type]["tokenizer"]
        model_class = MODEL_CLASSES[self.config.model_type]['backbone']

        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            #finetuning_task=config.task_name,
            cache_dir=None,
            use_cache=False,
        )

        self.tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)  # type: PreTrainedTokenizer

        if config.lora_r> 0:
            self.llm_model = model_class.from_pretrained(config.model_name_or_path, device_map="auto",torch_dtype = torch.bfloat16)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=self.config.lora_r, lora_alpha=self.config.lora_a, target_modules=target_modules, lora_dropout=0.01)

            self.model = get_peft_model(self.llm_model, peft_config)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.print_trainable_parameters()
        else:
            self.model = model_class.from_pretrained(config.model_name_or_path, config=model_config)
            #self.model.print_trainable_parameters()

        self.preprocessor = Preprocessor(self, self.config.model_type)

        self.init_projector()

    
    def init_projector(self):
        num_classes = 6                
        #  projector
        self.encoder = nn.Sequential(nn.Linear(self.config.encoder_output_dim, self.config.hidden_dim, dtype=torch.float), 
                                        nn.GELU(),
                                        nn.LayerNorm(self.config.hidden_dim, dtype=torch.float),
                                        nn.Dropout(p=self.config.encoder_dropout),
                                        nn.Linear(self.config.hidden_dim, self.config.emb_dim, dtype=torch.float))
 


        self.classifier = nn.Linear(self.config.emb_dim, num_classes,dtype=torch.float)
        
        self.loss_c = nn.NLLLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.l2 = self.config.l2
        self.t2 = self.config.t2
        if not self.t2: self.t2 = 1

        self.l3 = self.config.l3
        self.t3 = self.config.t3
        if not self.t3: self.t3 = 1


    @classmethod
    def from_pretrained(cls, path: str):
        """Load a pretrained wrapper from a given path."""
        wrapper = Citss.__new__(Citss)
        super(Citss, wrapper).__init__()
        
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]["tokenizer"]
        model_class = MODEL_CLASSES[wrapper.config.model_type]['backbone']


        if wrapper.config.lora_r > 0:
            wrapper.tokenizer = tokenizer_class.from_pretrained(wrapper.config.model_name_or_path)
            llm_model = model_class.from_pretrained(wrapper.config.model_name_or_path, device_map="auto",torch_dtype = torch.bfloat16)
            wrapper.model = PeftModel.from_pretrained(llm_model, path)
            wrapper.tokenizer.pad_token_id = wrapper.tokenizer.eos_token_id
        else:
            wrapper.tokenizer = tokenizer_class.from_pretrained(wrapper.config.model_name_or_path)
            wrapper.model = model_class.from_pretrained(path)

        wrapper.preprocessor = Preprocessor(wrapper, wrapper.config.model_type)
        wrapper.init_projector()
        wrapper.encoder.load_state_dict(torch.load(f'{path}/encoder.pt', weights_only=True))
        wrapper.classifier.load_state_dict(torch.load(f'{path}/classifier.pt', weights_only=True))

        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

        torch.save(self.encoder.state_dict(), f'{path}/encoder.pt')
        torch.save(self.classifier.state_dict(), f'{path}/classifier.pt')
        return
        

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), "r") as f:
            return jsonpickle.decode(f.read())

    def train(
            self,
            task_train_data: List[InputAugExample],
            device,
            per_gpu_train_batch_size: int = 8,
            num_train_epochs: int = 3,
            gradient_accumulation_steps: int = 1,
            weight_decay: float = 0.0,
            learning_rate: float = 5e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps=0,
            max_grad_norm: float = 1,
            logging_steps: int = -1,
            logging_number: int = 5,
            output_dir=None,
            eval_kwargs=None,
            **_
    ):
        if num_train_epochs == 0: num_train_epochs = 1

        N = len(task_train_data)
        t_total = N // gradient_accumulation_steps * num_train_epochs

        if logging_steps < 0: logging_steps = t_total // logging_number

        no_decay = ["bias", "LayerNorm.weight"] # these modules are not L2 regularized
       
        if self.config.lora_r > 0:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],"weight_decay": weight_decay,},
                {"params": [p for n, p in self.encoder.named_parameters() if not 'bias' in n] + [p for n, p in self.classifier.named_parameters() if not 'bias' in n],
                "weight_decay": weight_decay},
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad] \
                        + [p for n, p in self.encoder.named_parameters() if 'bias' in n] + [p for n, p in self.classifier.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,}
            ]
        else:
            optimizer_grouped_parameters = [
                { "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay,},
                {"params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)] + [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": weight_decay,},
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)] + [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)] + [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

     
        batch_step = 0
        accumulated_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_score = -1
        self.set_zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        for epoch in train_iterator:
            sample_index = [_ for _ in  range(N)]
            random.shuffle(sample_index)
            batch_start = 0
            batch_size = per_gpu_train_batch_size

            for step, batch_start in enumerate(tqdm(range(0, N, batch_size), desc="Batch")):
                batch_sample = sample_index[batch_start: batch_start+batch_size]
                batch = [task_train_data[_] for _ in batch_sample]
                self.set_train()
    
                if self.config.model_type=='llama':
                    batch_a, mask_s, mask_k, labels = self._convert_llama_examples_to_features(batch, eval=False, epoch=epoch)
                    loss = self.train_step(batch_a, mask_s, mask_k, labels, device)
                else:
                    batch_a, mask_s, mask_k, labels = self._convert_examples_to_features(batch, eval=False, epoch=epoch, device=device)
                    loss = self.train_step(batch_a,  mask_s, mask_k, labels, device)
                    
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if batch_step % gradient_accumulation_steps == 0:
                    self.clip_grad(max_grad_norm)
                    optimizer.step()
                    torch.cuda.empty_cache()
                    scheduler.step()
                    self.set_zero_grad()

                    accumulated_step += 1
                    if logging_steps > 0 and accumulated_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        if eval_kwargs is not None:
                            score = self.in_training_eval(eval_kwargs)
                            if score > best_score:
                                logger.info(f"New best score {score} > {best_score} previously, saving model")
                                self.save(output_dir)
                                best_score = score
                            else:
                                logger.info(f"Old best score {best_score} >= {score} currently, not saving model")

                        print(json.dumps({**logs, **{"step": accumulated_step}}))
                
                batch_step += 1

            logs = {}
            loss_scalar = (tr_loss - logging_loss) / logging_steps
            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar
            logs["loss"] = loss_scalar
            logging_loss = tr_loss

            if eval_kwargs is not None:
                score = self.in_training_eval(eval_kwargs)
                if score > best_score:
                    logger.info(f"New best score {score} > {best_score} previously, saving model")
                    self.save(output_dir)
                    best_score = score
                else:
                    logger.info(f"Old best score {best_score} >= {score} currently, not saving model")

            print(json.dumps({**logs, **{"step": accumulated_step}})) 

        if eval_kwargs is not None:
            score = self.in_training_eval(eval_kwargs)
            if score > best_score:
                logger.info(f"New best score {score} > {best_score} previously, saving model")
                self.save(output_dir)
            else:
                logger.info(f"Old best score {best_score} >= {score} currently, not saving model")

        return accumulated_step, (tr_loss / accumulated_step if accumulated_step > 0 else -1)

    def eval(
            self,
            eval_data: List[InputAugExample],
            device,
            per_gpu_eval_batch_size: int = 8,
            **_
    ) -> Dict:
        eval_batch_size = per_gpu_eval_batch_size
        N = len(eval_data)
        llm_device = self.model.device
        if self.config.model_type =='scibert':
            eval_dataset = self._generate_dataset(eval_data)
            eval_sampler = SequentialSampler(eval_dataset) 
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
            preds = None
            out_label_ids = None
            uids = None

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                self.set_eval()
    
                labels = batch["labels"].to(device)
                batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
                with torch.no_grad():
                    logits = self.eval_step(batch, device)
                if preds is None:
                    preds = logits
                    out_label_ids = labels
                else:
                    preds = torch.cat((preds, logits), dim=0)
                    out_label_ids = torch.cat((out_label_ids, labels), dim=0)
            
        else:
            preds = None
            out_label_ids = None
            for step, batch_start in enumerate(tqdm(range(0, N, eval_batch_size), desc="Batch")):
                batch = eval_data[batch_start: batch_start+eval_batch_size]
                self.set_eval()
    
                with torch.no_grad():
                    batch_a, labels = self._convert_llama_examples_to_features(batch, eval=True)
                    logits = self.eval_step(batch_a, device)
                if preds is None:
                    preds = logits
                    out_label_ids = labels
                else:
                    preds = torch.cat((preds, logits), dim=0)
                    out_label_ids = torch.cat((out_label_ids, labels), dim=0)

        preds = preds.detach().cpu().numpy()
        out_label_ids = out_label_ids.detach().cpu().numpy()
        return {"logits": preds, "labels": out_label_ids}


    def in_training_eval(self, eval_kwargs):
        eval_results = self.eval(**eval_kwargs)
        predictions = np.argmax(eval_results["logits"], axis=1)

        mif = f1_score(eval_results["labels"], predictions, average='micro')
        maf = f1_score(eval_results["labels"], predictions, average="macro")
        
        # validation criteria
        return maf + mif

    
    def set_eval(self):
        self.model.eval()
        self.encoder.eval()
        self.classifier.eval()

    def set_train(self):
        self.model.train()
        self.encoder.train()
        self.classifier.train()

    def set_zero_grad(self):
        self.model.zero_grad()
        self.encoder.zero_grad()
        self.classifier.zero_grad()

    def clip_grad(self, max_grad_norm):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_grad_norm)

    def _generate_dataset(self, data: List[InputAugExample], labelled: bool = True):
        features = self._convert_eval_examples_to_features(data, labelled=labelled)
        feature_dict = {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.bool),
            "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            "labels": torch.tensor([f.label for f in features], dtype=torch.long),
            "mlm_labels": torch.tensor([f.mlm_labels for f in features], dtype=torch.long),
        }

        return DictDataset(**feature_dict)

    def _convert_eval_examples_to_features(
            self, examples: List[InputAugExample], labelled: bool = True
    ) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 100000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.preprocessor.get_input_features(example, labelled=labelled)
            features.append(input_features)
        return features

    def _convert_examples_to_features(self, examples: List[InputAugExample], eval=False, epoch=-1, device=None) -> List[InputFeatures]:
        features_a = {'input_ids':[], 'attention_mask':[]}
        features_s = {'input_ids':[], 'attention_mask':[]}
        features_k = {'input_ids':[], 'attention_mask':[]}

        mask_s = torch.zeros(size=(len(examples),), dtype=bool)
        mask_k = torch.zeros(size=(len(examples),), dtype=bool)

        for i, example in enumerate(examples):
            input_feature_a = self.preprocessor.get_input_aug_features(InputAugExample(example.guid, text_a=example.text_a,label=example.label))
            features_a['input_ids'].append(input_feature_a.input_ids)
            features_a['attention_mask'].append(input_feature_a.attention_mask)
            
            if not eval:
                n = len(example.text_s)
                if n >= 1: 
                    mask_s[i] = 1
                    input_feature_s = self.preprocessor.get_input_aug_features(InputAugExample(example.guid, text_a=example.text_s[int(epoch%n)], label=example.label))
                    features_s['input_ids'].append(input_feature_s.input_ids)
                    features_s['attention_mask'].append(input_feature_s.attention_mask)
                  
                n = len(example.text_k)
                if n >= 1:
                    mask_k[i] = 1
                    input_feature_k = self.preprocessor.get_input_aug_features(InputAugExample(example.guid, text_a=example.text_k[int(epoch%n)], label=example.label))
                    features_k['input_ids'].append(input_feature_k.input_ids)
                    features_k['attention_mask'].append(input_feature_k.attention_mask)
                      

        features_a['input_ids'] = torch.tensor(features_a['input_ids']+features_s['input_ids']+features_k['input_ids'], dtype=torch.long).to(device)
        features_a['attention_mask'] = torch.tensor(features_a['attention_mask']+features_s['attention_mask']+features_k['attention_mask'], dtype=torch.long).to(device)
        labels =  torch.tensor([int(e.label) for e in examples], dtype=torch.long).to(device)
        
        if not eval: return features_a, mask_s.to(device), mask_k.to(device), labels
        else: return features_a,  labels


    def _convert_llama_examples_to_features(self, examples: List[InputAugExample],  eval=False, epoch=-1) -> List[InputFeatures]:
        max_length = 750

        text_a = []
        text_s = []
        text_k = []
        if not eval:
            mask_s = torch.zeros(size=(len(examples),), dtype=bool)
            mask_k = torch.zeros(size=(len(examples),), dtype=bool)

        for i, e in enumerate(examples):
            text_a.append(e.text_a)
            
            if not eval:
                n = len(e.text_s)
                if n >= 1: 
                    mask_s[i] = 1 
                    if n == 1: text_s.append(e.text_s[0])
                    else: text_s.append(e.text_s[int(epoch%n)])

                n = len(e.text_k)
                if n>=1:
                    mask_k[i] = 1
                    if n == 1: text_k.append(e.text_k[0])
                    else:
                        text_k.append(e.text_k[int(epoch%n)])
        
        labels = torch.tensor([int(e.label) for e in examples], dtype=torch.long)
        if not eval:
            batch_a = self.tokenizer(text_a+text_s+text_k, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            return batch_a, mask_s, mask_k, labels
        else:
            batch_a = self.tokenizer(text_a, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            return batch_a, labels


    def train_step(self, labeled_batch, mask_s, mask_k, labels, device) -> torch.Tensor:        
        # Classification loss
        llm_device = self.model.device
        N0 = len(labels)
        N1 = mask_s.sum()
        N2 = mask_k.sum()
  
        z, x = self.compute_emb(labeled_batch, llm_device, device)   
        z0 = z[:N0]
        z1 = z[N0:N0+N1]
        z2 = z[N0+N1:]

        logits = self.classifier(z0)
        labels = labels.to(device)
        loss_c = self.supervised_loss(logits, labels.view(-1))
        
        # Self-supervised loss
        if self.l2>0 and N1>=1: 
            loss_s = self.contrastive_loss(z0[mask_s.to(device)], z1, tau=self.t2)
        else: loss_s  = self.assign_zero(device)

        if self.l3>0 and N2>=1: 
            loss_k = self.contrastive_loss(z0[mask_k.to(device)], z2, tau=self.t3)
        else: loss_k  = self.assign_zero(device)
        
        loss = loss_c + self.l2 * loss_s + self.l3 * loss_k
        return loss


    def compute_emb(self, batch_s, llm_device, device):
        if isinstance(batch_s, dict):
            last_hidden_state = self.model(**batch_s, output_hidden_states=True).hidden_states[-1]
            sequence_lengths = batch_s["attention_mask"].sum(dim=1) - 3 
            x_s = last_hidden_state[torch.arange(len(sequence_lengths), device=device), sequence_lengths]
            return self.encoder(x_s), False
        else:
            last_hidden_state = self.model(**(batch_s.to(llm_device))).last_hidden_state
            last_hidden_state = last_hidden_state.to(device)
            sequence_lengths = batch_s["attention_mask"].sum(dim=1) - 1 
            x_s = last_hidden_state[torch.arange(len(sequence_lengths), device=device), sequence_lengths.to(device)]
            return self.encoder(x_s.float()), False

    def eval_step(self, batch: Dict[str, torch.Tensor], device) -> torch.Tensor:
        llm_device = self.model.device
        z, _ = self.compute_emb(batch, llm_device, device)
        logits = self.classifier(z) 
        return logits
    

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor, mean: bool = True, tau=1.0, mask=None):
        if isinstance(mask, torch.Tensor): between_sim = self.sim(h1[mask], h2[mask])/tau
        else: between_sim = self.sim(h1, h2)/tau
        # numerical stable 
        ret = -self.softmax(between_sim).diag()
        ret = ret.mean() if mean else ret.sum()
        return ret
    
    
    def supervised_loss(self, logits, y):
        y1 = self.softmax(logits) 
        return self.loss_c(y1, y)

    def assign_zero(self, device):
        return torch.zeros((1,)).to(device)
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.mm(z1, z2.t())



