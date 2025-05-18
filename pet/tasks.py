"""
This file contains the logic for loading training and test data for all tasks.
"""
import json
import pandas as pd
import pet.log as log

from pet.utils import InputAugExample
from pet.augment_stk import KP

logger = log.get_logger("root")

# EOL prompt
prompt_template = '''You are provided a context from a paper P citing a paper Q, with the specific citation marked as the '#CITATION_TAG' tag. Please analyze the citation function of the context, which represents the author’s motive or purpose for citing Q. 
Here is the context: "{}". Only output one word as the answer:'''

class CitssProcessor:
    def __init__(self, data_dir, use_eol, use_sc, beta, gamma, kp_mode):
        self.data_dir = data_dir
        self.use_eol = use_eol
        self.use_sc = use_sc
        self.beta = beta
        self.gamma = gamma
        self.kp_mode = kp_mode


    def create_examples(self, dataset, set_type):
        examples = []
        data = pd.read_csv(f"{self.data_dir}/{dataset}_{set_type}.txt", sep='\t')

        if set_type!='train':

            for i, line in data.iterrows():
                uid = str(line['unique_id'])
                guid = "%s-%s" % (set_type, uid)

                raw_input = line['input_context']

                if self.use_eol: text_a = prompt_template.format(raw_input)
                else: text_a = str(raw_input)
                
                label = str(line['label'])
                
                examples.append(
                    InputAugExample(guid=guid, text_a=text_a, label=label))
        else:
            if self.use_sc:
                with open(f"{self.data_dir}/{dataset}_sas.json", 'r') as fp: sc_data = json.load(fp)
            else: sc_data = {}

            kp_data = {}
            if self.beta>=0: kp_data = KP(dataset, 10, beta=self.beta, gamma=self.gamma, all_mode=self.kp_mode, dump=True)
            else: kp_data = {}


            for i, line in data.iterrows():
                uid = str(line['unique_id'])
                guid = "%s-%s" % (set_type, uid)

                raw_input = line['input_context']
                if self.use_eol: text_a = prompt_template.format(raw_input)
                else: text_a = str(raw_input)
                label = str(line['label'])

                if uid in sc_data:
                    if self.use_eol: text_s = [prompt_template.format(_) for _ in sc_data[uid]]
                    else: text_s = sc_data[uid]
                else: text_s = []

                if uid in kp_data:
                    if self.use_eol: text_k = [prompt_template.format(_) for _ in kp_data[uid]]
                    else: text_k = kp_data[uid]
                else: text_k = []

                examples.append(
                    InputAugExample(guid=guid, text_a=text_a, text_s=text_s, text_k=text_k,  label=label))
             
        return examples
    
    

