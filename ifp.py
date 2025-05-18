import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" 
from sklearn.metrics import f1_score
import transformers
from tqdm import tqdm
import pandas as pd
import re

def fix_seed(seed):
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

LABEL_MAP = {"BACKGROUND": 0, "COMPARES_CONTRASTS": 1, "EXTENSION":2, "FUTURE":3,
              "MOTIVATION": 4, "USES": 5}

def verify_predictions(pred):
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, pred)
    try:
        if match:
            res = match.group(1).strip().upper()
            if res in LABEL_MAP:
                res = LABEL_MAP[res]
                return res
    except:
        pass
    print('invalid output\t', pred)
    return 6                 
            

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset", type=str, default="acl_arc", choices=["act2", "acl_arc", "focal"])
    parser.add_argument("--llm_dir", type=str, default='llama/meta-llama/Meta-Llama-3-8B-Instruct',)
    parser.add_argument("--random_seed", type=int, default=2024)

    args = parser.parse_args()
    return args

################################################################################################
# Prompts
################################################################################################
system_text = {"role": "system", "content": "You are good at reading academic papers." }

prompt_template = '''
You are provided a context from a paper P citing a paper Q, with the specific citation marked as the '#CITATION_TAG' tag. Please analyze the citation function of the context, which represents the author’s motive or purpose for citing Q. The six classes of citation functions are: 
- [BACKGROUND]: The cited paper Q provides relevant information or is part of the body of literature in this domain.
- [COMPARES_CONTRASTS]: The citing paper P expresses similarities or differences to, or disagrees with the cited paper Q.
- [EXTENSION]: The citing paper P extends the data, methods, etc. of the cited paper Q.
- [FUTURE]: The cited paper Q is a potential avenue for future work.
- [MOTIVATION]: The citing paper P is directly motivated by the cited paper Q.
- [USES]: The citing paper P uses the methodology or tools created by the cited paper Q. 

Here is the context: "{}"
Only output the most appropriate class to categorize #CITATION_TAG and enclose the label within square brackets [].
'''

REVERSE_LABEL_MAP = {'0': "BACKGROUND", '1': "COMPARES_CONTRASTS", '2': "EXTENSION", '3':"FUTURE",
              '4':"MOTIVATION", '5': "USES"}


def compose_prompt(texts):
    return [[system_text, {"role": "user", "content": prompt_template.format(text)}] for text in texts]      

def run_with_suffix(args, pipeline=None):
    print("setup data loader ...")
    terminators = [pipeline.tokenizer.eos_token_id,pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    predicted = []

    output_file_test = os.path.join(args.output_dir, str(args.dataset)+"_"+str(args.model)+"_"+f'output_test.csv')
    results_file = os.path.join(args.output_dir, str(args.dataset)+"_"+str(args.model)+"_"+f'results.txt')
    
    with torch.no_grad():
        data = pd.read_csv(f"/data/{args.dataset}_test.txt", sep='\t')
        N = len(data)
        batch_size = 4
        end_id = 0
        for i in tqdm(range(0, N, batch_size)):
            end_id = min(N,i+batch_size)
            texts = [data.iloc[lineid]['input_context'] for lineid in range(i, end_id)]
            messages = compose_prompt(texts)
            
            prompts = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipeline(prompts, max_new_tokens=64, eos_token_id=terminators,
                    do_sample=True, temperature=0.6, top_p=0.9, pad_token_id = pipeline.tokenizer.eos_token_id) 
            
        
            responses = [outputs[j][0]["generated_text"][len(prompt):] for j, prompt in enumerate(prompts)]
            predicted += [verify_predictions(r) for r in responses]



        
    labels = data['label'].tolist()
    label_range = [0,1,2,3,4,5]

    output_data = pd.DataFrame({'unqiue_id':data['unique_id'][:end_id], 'predicted':predicted, 'label':labels[:end_id]})
    output_data.to_csv(output_file_test,sep='\t',index=False)

    macro_f1 = f1_score(labels[:end_id], predicted, average='macro', labels=label_range)
    micro_f1 = f1_score(labels[:end_id], predicted, average='micro', labels=label_range)
    with open(results_file, 'a') as f:

        f.write("Testdata Results:"+'\n')
        f.write("Macro F-Score: {}".format(macro_f1)+'\n')
        f.write("Micro F-Score: {}".format(micro_f1)+'\n')    


if __name__ == "__main__":
    args = parse_arguments()
    fix_seed(args.random_seed)

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.llm_dir,
        model_kwargs={'torch_dtype':torch.bfloat16},
        device_map="auto")
  
    if not os.path.exists(os.path.join(args.output_dir)):
        os.makedirs(os.path.join(args.output_dir))
        print("Directory created!")
    else: print("Directory already exists!")

    run_with_suffix(args, pipeline=pipeline)
 


