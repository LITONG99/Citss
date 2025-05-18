import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["WANDB_DISABLED"]="true"

import torch
import pet
from pet.tasks import CitssProcessor
from pet.wrapper import WrapperConfig

def finetuning(model_type, l2, l3,  t2, t3, dataset, lora_r, lora_a, use_eol, beta, gamma=0.1, kp_mode=['gr','ab','lr']):
    '''
    Input:
        - model_type: 'llama' for Meta-Llama-3-8B-Instruct, 'scibert' for SciBERT. 
        - l2, l3, t2, t3 are the \lambdas and temperatures for SC and KP in the paper.
        - use eol: set True for instruction-tuned LLMs. 
        - beta, gamma: the perturbation probability for STKs and synonyms. 
        - kp_mode: KP operation.
    '''

    # set the hyperparameters
    if model_type == 'llama':
        pet_per_gpu_train_batch_size = 4
        pet_per_gpu_eval_batch_size = 8
        
        if dataset == 'acl_arc':
            logging_steps = 800
            pet_num_train_epochs = 10
            hidden_dim = 1024
            emb_dim = 256
        elif dataset == 'focal': # focal
            logging_steps = 1600
            pet_num_train_epochs = 8
            hidden_dim = 1024
            emb_dim = 256
        else:
            logging_steps = 1600
            pet_num_train_epochs = 8
            hidden_dim = 128
            emb_dim= 64

        model_name_or_path = "../meta-llama/Meta-Llama-3-8B-Instruct" # llama-8B path
        encoder_dropout=0.05
        pet_max_seq_length = 1024
        encoder_output_dim = 4096
    else:
        if dataset == 'acl_arc':
            pet_per_gpu_train_batch_size = 4
            pet_num_train_epochs = 8
            encoder_dropout=0.05
            hidden_dim = 1024
            emb_dim = 256
        elif dataset == 'act2':
            pet_per_gpu_train_batch_size = 4
            pet_num_train_epochs = 8
            encoder_dropout=0 
            hidden_dim = 256
            emb_dim = 128
        else: # focal
            pet_per_gpu_train_batch_size = 16
            pet_num_train_epochs = 8
            encoder_dropout=0
            hidden_dim = 256
            emb_dim = 128


        pet_per_gpu_eval_batch_size = 16
        model_name_or_path = '../scibert' # scibert path
        pet_max_seq_length = 512
        encoder_output_dim = 768

    
    n_gpu = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else "cpu"


    # load data 
    data_dir = f'data/'
    processor = CitssProcessor(data_dir, use_eol, use_sc = l2!=0, beta = -1 if l3==0 else beta,  gamma=gamma, kp_mode=kp_mode)
    train_data = processor.create_examples(dataset, 'train')
    dev_data = processor.create_examples(dataset, 'valid')
    test_data = processor.create_examples(dataset, 'test')

    output_dir = f"output/{dataset}-{model_type}"
    os.makedirs(output_dir, exist_ok=True)

    model_cfg  = WrapperConfig(
        model_type=model_type,
        model_name_or_path=model_name_or_path,
        label_list=["0", "1", "2", "3", "4", "5"],
        max_seq_length=pet_max_seq_length,
        l2=l2, l3=l3, 
        t2=t2, t3=t3, 
        dataset=dataset,
        lora_r=lora_r,
        lora_a=lora_a,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        encoder_dropout= encoder_dropout,
        encoder_output_dim = encoder_output_dim,
    )

    train_cfg = pet.TrainConfig(
        device=device,
        n_gpu = n_gpu,
        per_gpu_train_batch_size=pet_per_gpu_train_batch_size,
        num_train_epochs=pet_num_train_epochs,
        gradient_accumulation_steps=1,
        weight_decay=0.01, # \omega
        learning_rate=2e-5,
        adam_epsilon=1e-8,
        warmup_steps=50,
        max_grad_norm=1.0,
        logging_number=-1,
    )

    eval_cfg =  pet.EvalConfig(
        device=device,
        n_gpu=n_gpu,
        metrics= ["f1-macro", "f1-micro"],
        per_gpu_eval_batch_size=pet_per_gpu_eval_batch_size,
    )

    final_results = pet.train_citss(
        model_cfg,
        train_cfg,
        eval_cfg,
        output_dir=output_dir,
        repetitions=3,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data,
        do_train=True,
        do_eval=True,
        overwrite_dir=True,
        save_model=False,
        from_pretrained = ''# you can use a trained checkpoint 
    )


if __name__ == "__main__":
    # finetune scibert
    finetuning('scibert', l2=0.2, l3=0.1,t2=1, t3=1, dataset='acl_arc', lora_r=-1, lora_a=-1, use_eol=False, beta=0.6)

    # finetune Llama3-8B
    #finetuning('llama',  0.1, 0.2, 1, 1, 'acl_arc', lora_r = 16, lora_a = 16, use_eol=True, beta=0.4)




