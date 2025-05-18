import json
import os
import statistics
import warnings
from abc import ABC
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import torch
from sklearn.metrics import f1_score

from pet.utils import InputAugExample, save_logits, save_predictions,set_seed
from pet.wrapper import WrapperConfig, Citss



class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, "r", encoding="utf8") as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(
        self,
        device: str = None,
        n_gpu: int = 0,
        per_gpu_train_batch_size: int = 8,
        num_train_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        logging_number: int = 5,
        logging_steps: int = -1,
        max_grad_norm: float = 1,
    ):

        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.logging_number = logging_number
        self.logging_steps = logging_steps
        self.max_grad_norm = max_grad_norm



class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(
        self,
        device: str = None,
        n_gpu: int = 1,
        per_gpu_eval_batch_size: int = 8,
        metrics: List[str] = None,
        local_rank=-1,
    ):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        """
        if local_rank != -1: self.device = local_rank
        else: self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
  


def train_citss(
    model_config: WrapperConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    output_dir: str,
    repetitions: int = 3,
    train_data: List[InputAugExample] = None,
    dev_data: List[InputAugExample] = None,
    test_data: List[InputAugExample] = None,
    do_train: bool = True,
    do_eval: bool = True,
    seed: int = 42,
    overwrite_dir: bool = False,
    save_model=False,
    from_pretrained: str = '' 
):
    results = defaultdict(lambda: defaultdict(list))
    
    set_seed(seed)
    
    for iteration in range(0, repetitions):
        results_dict = {}
        pattern_iter_output_dir = "{}/proj-i{}".format(output_dir, iteration)

        if os.path.exists(pattern_iter_output_dir) and not overwrite_dir:
            print(f"Path {pattern_iter_output_dir} already exists, skipping it...")
            continue

        if not os.path.exists(pattern_iter_output_dir):
            os.makedirs(pattern_iter_output_dir)

        
        # Training
        if do_train:
            if len(from_pretrained)>0:
                wrapper = Citss.from_pretrained(from_pretrained)
            else:
                wrapper = Citss(model_config)

            results_dict.update(
                train_single_citss(
                    wrapper,
                    train_data,
                    train_config,
                    pattern_iter_output_dir,
                    dev_data,
                    eval_config,
                )
            )

            with open(os.path.join(pattern_iter_output_dir, "results.txt"), "w") as fh:
                fh.write(str(results_dict))

            print("Saving trained model at {}...".format(pattern_iter_output_dir))
            train_config.save(os.path.join(pattern_iter_output_dir, "train_config.json"))
            eval_config.save(os.path.join(pattern_iter_output_dir, "eval_config.json"))
            print("Saving complete")

            try:
                wrapper.model = None
                wrapper = None
            except:
                pass
            torch.cuda.empty_cache()


        # Evaluation
        if do_eval:
            print("Starting evaluation...")
            try:
                wrapper = Citss.from_pretrained(pattern_iter_output_dir)
            except OSError:
                warnings.warn("No model found saved, proceeding with current model instead of best")
                pass

            for split, eval_data in {"dev": dev_data, "test": test_data}.items():
                if eval_data is None:
                    continue
                eval_result = evaluate_citss(wrapper, eval_data, eval_config)
                

                save_predictions(
                    os.path.join(pattern_iter_output_dir, "predictions.jsonl"), wrapper, eval_result
                )
                if 'logits' in eval_result:
                    save_logits(os.path.join(pattern_iter_output_dir, "eval_logits.txt"), eval_result["logits"])


                scores = eval_result["scores"]
                print("--- {} result (iteration={}) ---".format(split, iteration))
                print(scores)

                results_dict[f"{split}_set_after_training"] = scores
                with open(os.path.join(pattern_iter_output_dir, "results.json"), "w") as fh:
                    json.dump(results_dict, fh)

                for metric, value in scores.items():
                    results[split][metric].append(value)

            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()


        if do_train and not save_model:
            outputs = os.listdir(pattern_iter_output_dir)
            for item in outputs:
                if item.endswith(".bin") or item.endswith("tensors") or item.endswith(".pt"):
                    os.remove(os.path.join(pattern_iter_output_dir, item))
    if do_eval:
        print("=== OVERALL RESULTS ===")
        results_to_log = _write_results(os.path.join(output_dir, "result_test.txt"), results)
    else:
        print("=== ENSEMBLE TRAINING COMPLETE ===")
        results_to_log = None

    return results_to_log


def train_single_citss(
    model,
    train_data: List[InputAugExample],
    config: TrainConfig,
    output_dir,
    dev_data: List[InputAugExample] = None,
    eval_config: EvalConfig = None,
):

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    results_dict = {}

    if model.config.model_type=='scibert':
        model.model.to(device)
    model.encoder.to(device)
    model.classifier.to(device)


    if dev_data is not None and eval_config is not None:
        eval_kwargs = {
            "eval_data": dev_data,
            "device": device,
            "per_gpu_eval_batch_size": eval_config.per_gpu_eval_batch_size,
            "n_gpu": eval_config.n_gpu,
            "metrics": eval_config.metrics,
        }
    else:
        eval_kwargs = None

    if not train_data:
        print("Training method was called without training examples")
    else:
        global_step, tr_loss = model.train(
            train_data,
            device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu = config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            logging_steps=config.logging_steps,
            logging_number=config.logging_number,
            output_dir=output_dir,
            eval_kwargs=eval_kwargs,
        )
        results_dict["global_step"] = global_step
        results_dict["average_loss"] = tr_loss

    return results_dict


def evaluate_citss(
    model,
    eval_data: List[InputAugExample],
    config: EvalConfig,
) -> Dict:
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    if model.config.model_type=='scibert':
        model.model.to(device)

    model.encoder.to(device)
    model.classifier.to(device)
    results = model.eval(
        eval_data,
        device,
        per_gpu_eval_batch_size=config.per_gpu_eval_batch_size)

    predictions = np.argmax(results["logits"], axis=1)
    scores = {}
    scores["f1-macro"] = f1_score(results["labels"], predictions, average="macro")
    scores["f1-micro"] = f1_score(results["labels"], predictions, average="micro")
 
    results["scores"] = scores
    results["predictions"] = predictions
    results['uid'] = [tmp.guid for tmp in eval_data]

    return results


def _write_results(path: str, results: Dict) -> Dict:
    final_results_dict = {}

    with open(path, "a") as fh:
        for split, split_results in results.items():
            for metric in split_results.keys():
                values =  split_results[metric]
                all_results = "{}_{}:{}".format(split, metric, '\t'.join([str(round(_, 4)) for _ in values]))
                fh.write(all_results + "\n")
                final_results_dict[f"{split}-{metric}"] = np.mean(values)

    return final_results_dict
