import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 

import transformers
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset
import multiprocessing
import pandas as pd

def fix_seed(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class QADataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        data_dir = f'data/{dataset}_train.txt'
        data = pd.read_csv(data_dir, sep='\t')
        self.unique_ids = data['unique_id']
        self.context = data['input_context']
        self.len = len(self.unique_ids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.unique_ids[index], self.context[index])

    

def setup_llama_data_loader(dataset, random_seed=2024):
    fix_seed(random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_seed):
        np.random.seed(worker_seed)
        random.seed(worker_seed) 

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, 4)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = QADataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             batch_size=1,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True)

    return dataloader

################################################################################################
# Prompts
################################################################################################
system_text = {"role": "system", "content": "You are good at reading academic papers." }

prompt_description_stk = '''
You are provided a context from a paper P, and please ignore the #CITATION_TAG. Your task is to identify scientific keyphrases from the context. Each scientific keyphrase belongs to one of the following classes:
- [Task]: The scientific problem or research focus addressed in the paper. It outlines the specific objectives or questions that the study aims to answer.  
- [Material]: All materials utilized in the study, such as experimental tools, datasets, and the objects or subjects of investigation. It details the resources of the research.
- [Technique]: The specific methods, models, frameworks, or systems. It identifies the approaches taken to analyze data or solve problems.
- [Process]: It describes a sequence of steps or operations involved in a particular procedure, algorithm, or workflow. It emphasizes the procedural aspects.
- [Measure]: This class pertains to the metrics, indicators, or criteria used to assess or quantify the outcomes of the study.
- [Concept]: This category encompasses scientific keyphrases that do not fit into the aforementioned classes. It may include phenomena, theoretical terms, or entities relevant to the field of study.
Output your answer only in JSON format and be consistent with the text in the original context. Specifically, if there is any keyphrase of a certain class, use the class label as the key and the list of keyphrases as the value. 

Here is an example: "The framework represents a generalization of several predecessor NLG systems based on Meaning-Text Theory: FoG (Kittredge and Polgu~re, 1991), LFS (Iordanskaja et al., 1992), and JOYCE (Rambow and Korelsky, 1992). The framework was originally developed for the realization of deep-syntactic structures in NLG ( #CITATION_TAG )"
Output:{'Technique': ['NLG systems', 'FoG', 'LFS', 'JOYCE', 'Meaning-Text Theory'], 'Concept':['deep-syntactic structures']}
'''
prompt_context = 'Here is the context: "{}"'

def compose_stk_prompt(context):
    res = prompt_description_stk
    res += prompt_context.format(context)
    return [system_text, {"role": "user", "content": res}] 

def main_stk(dataset, llm_dir='llama/meta-llama/Meta-Llama-3-70B-Instruct', output_dir='stk'):
    pipeline = transformers.pipeline(
        "text-generation",
        model=llm_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto")


    output_file_test = os.path.join(output_dir, f"{dataset}_raw_stk.csv" )

    dataloader = setup_llama_data_loader(dataset)
    terminators = [pipeline.tokenizer.eos_token_id,pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    uids = []
    pred = []
    for i, data in tqdm(enumerate(dataloader)): # you can batch this loop to accelerate if you have more GPU mem
        uid, sides, _,  _,  _ = data
        uids.append(uid[0])
        message = compose_stk_prompt(sides={k:sides[k]for k in sides})
        input = pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        output = pipeline(input, max_new_tokens=1024, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9, pad_token_id = pipeline.tokenizer.eos_token_id)
        pred.append(output[0]["generated_text"][len(input):])

        
    output_data = pd.DataFrame({'unqiue_id':uids, 'response':pred})
    output_data.to_csv(output_file_test, sep='\t',index=False)


fix_seed(2024)
main_stk('acl_arc')
main_stk('focal')
#main_stk('act2')



