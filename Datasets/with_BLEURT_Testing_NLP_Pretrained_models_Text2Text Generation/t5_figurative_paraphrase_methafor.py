#Description: The methaphor.csv file has 2 columns: premise and  hypotesis
#Sources:
#https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=figurative-nlp
#1.figurative-nlp/t5-figurative-paraphras:https://huggingface.co/figurative-nlp/t5-figurative-paraphrase?text=No%2C+I%27d+say+this+was+dumbest
#This model can convert the figurative/metaphorical expression to the literal expression.
#2.NLP(text2TextGeneration)

from bleurt import score
import numpy as np
scorer = score.BleurtScorer()


import pandas as pd
premise_df = pd.read_csv(r"C:\Users\adela\Workspace-Python\t5-figurative-paraphrase\metaphor.csv")
premise_df.loc[0].index
#print(premise_df.loc[0].index,"\n")
premise_df.premise.index
#print(premise_df.hyperbole.index,"\n") 
premise_df.hypothesis.index
#print(premise_df.literal_expression.index,"\n")
premise_column = premise_df["premise"]
#print(premise_column,"\n")
#Iterating over column
list_premise_column = list(premise_column)[:100]
#list_premise_column = premise_column.values

#print(list_premise_column[:3])
#for i in list_premise_column[:10]:
    #print(i)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "figurative-nlp/t5-figurative-paraphrase"
#model_name = "figurative-nlp/sarcasm-hyperbole-humor-paraphrase"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)



input_ids = tokenizer(list_premise_column, padding=True,
    truncation=False,
    max_length=512,return_tensors="pt").input_ids
print(input_ids)
outputs = model.generate(input_ids)
print("OUT: ", outputs)


counter_int = 0
candidates = []
for i in range(len(outputs)):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    candidates.append(str(result))
    print(i, "TRANSFORMER ", result)
    print(i, "METAPHORIC LANGUAGE", list_premise_column[i],"\n")
    if result != list_premise_column[i]:
        counter_int +=1

scores = scorer.score(candidates=candidates, references=list_premise_column)    

print('For figurative-nlp/sarcasm-hyperbole-humor-paraphrase model,the average BLEURT score for METHAFOR is {}'.format(np.mean(scores)))
#Get:For figurative-nlp/sarcasm-hyperbole-humor-paraphrase model,the average BLEURT score for METHAFOR is 0.6576590889692306

print("figurative-nlp/t5-figurative-paraphrase, out of a hundred sentences, only ",counter_int, " were modified by the transformer")#21  were modified by the transformer
#print("figurative-nlp/sarcasm-hyperbole-humor-paraphrase, out of a hundred sentences, only ",counter_int, " were modified by the transformer")#50  were modified by the transformer


#counter = {}
#counter[result] = 1
#if result == list_hyperbole_column[i]:
    #counter[result] = 0
#print(counter)    