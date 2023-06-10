#Irony
#From Wikipedia:This is the simplest form of irony, in which the speaker says the opposite of what he or she intends. 
#There are several forms, including euphemism, understatement, sarcasm, and some forms of humor. 

#Description: The sarcasm.csv file has 2 columns: premise and  hypotesis of the sarcasm
#Sources:
#https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=figurative-nlp
#1.figurative-nlp/sarcasm-hyperbole-humor-paraphras:https://huggingface.co/figurative-nlp/sarcasm-hyperbole-humor-paraphrase 
#No description about this model,I just tried to see what we get, even if we don't have the description of the model
#2.figurative-nlp/t5-figurative-paraphras:https://huggingface.co/figurative-nlp/t5-figurative-paraphrase?text=No%2C+I%27d+say+this+was+dumbest
#This model can convert the figurative/metaphorical expression to the literal expression.
#3.NLP(text2TextGeneration)


from bleurt import score
import numpy as np
scorer = score.BleurtScorer()


import pandas as pd
premise_df = pd.read_csv(r"C:\Users\adela\Workspace-Python\t5-figurative-paraphrase\sarcasm.csv")
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

#There are 2 models:figurative-nlp/sarcasm-hyperbole-humor-paraphrase and figurative-nlp/t5-figurative-paraphrase

#model_name = "figurative-nlp/sarcasm-hyperbole-humor-paraphrase"
model_name = "figurative-nlp/t5-figurative-paraphrase"

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
    print(i, "TRANSFORMER: ", result)
    print(i, "SARCASM LANGUAGE: ", list_premise_column[i],"\n")
    if result != list_premise_column[i]:
        counter_int +=1

scores = scorer.score(candidates=candidates, references=list_premise_column)

print('For figurative-nlp/t5-figurative-paraphrase model,the average BLEURT score for SARCASM is {}'.format(np.mean(scores)))
#Get:For figurative-nlp/t5-figurative-paraphrase model,the average BLEURT score for SARCASM is 0.16986634746193885    
#print('For figurative-nlp/sarcasm-hyperbole-humor-paraphrase model,the average BLEURT score for SARCASME is {}'.format(np.mean(scores)))
#Get:For figurative-nlp/sarcasm-hyperbole-humor-paraphrase model,the average BLEURT score for SARCASME is 0.258478859513998

#print("figurative-nlp/sarcasm-hyperbole-humor-paraphrase model for SARCASM, out of a total of 100 sentences, ",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/sarcasm-hyperbole-humor-paraphrase model for SARCASM, out of a total of 100 sentences,  74 sentences were modified by the transformer
print("figurative-nlp/t5-figurative-paraphrase model for SARCASM, out of a total of 100 sentences, ",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/t5-figurative-paraphrase model for SARCASM, out of a total of 100 sentences,  86 sentences were modified by the transformer