#Idioms is a figure of speech
#Idioms: An idiom is an expression that cannot be understood from the meanings of its separate words but that has a separate meaning of its own.
#Many (although not all) idioms are examples of figurative language.


#Description: The idioms.csv file has 2 columns:ideom and definition
#Sources:
#https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=figurative-nlp
#1.figurative-nlp/t5-figurative-paraphras:https://huggingface.co/figurative-nlp/t5-figurative-paraphrase?text=No%2C+I%27d+say+this+was+dumbest
#This model can convert the figurative/metaphorical expression to the literal expression.
#2.figurative-nlp/sarcasm-hyperbole-humor-paraphrase:https://huggingface.co/figurative-nlp/sarcasm-hyperbole-humor-paraphrase 
#There is no description about this pretained model in Hugging face
#3.NLP(text2TextGeneration)


import pandas as pd
idioms_df = pd.read_csv(r"C:\Users\adela\Workspace-Python\t5-figurative-paraphrase\idioms.csv")
idioms_df.loc[0].index
#print(idiom_df.loc[0].index,"\n")
idioms_df.idiom.index
#print(idioms_df.idiom.index,"\n") 
idioms_df.definition.index
#print(idioms_df.definition.index,"\n")
idiom_column = idioms_df["idiom"]
#print(idiom_column,"\n")
#Iterating over column
list_idioms_column = list(idiom_column)[:100]
#list_idiom_column = idiom_column.values

#print(list_idioms_column[:3])
#for i in list_idioms_column[:10]:
    #print(i)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#There are 2 model:figurative-nlp/t5-figurative-paraphrase and figurative-nlp/sarcasm-hyperbole-humor-paraphrase
model_name = "figurative-nlp/t5-figurative-paraphrase"
#model_name = "figurative-nlp/sarcasm-hyperbole-humor-paraphrase"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)



input_ids = tokenizer(list_idioms_column, padding=True,
    truncation=False,
    max_length=512,return_tensors="pt").input_ids
print(input_ids)
outputs = model.generate(input_ids)
print("OUT: ", outputs)


counter_int = 0
for i in range(len(outputs)):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(i, "TRANSFORMER: ", result)
    print(i, "IDIOM: ", list_idioms_column[i],"\n")
    if result != list_idioms_column[i]:
        counter_int +=1

print("figurative-nlp/t5-figurative-paraphrase model for IDIOMS, out of a total of 100 sentences,",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/t5-figurative-paraphrase model for IDIOMS, out of a total of 100 sentences, 100 sentences were modified by the transformer
#Remark:But if you analyze the text, it doesn't change anything
#print("figurative-nlp/sarcasm-hyperbole-humor-paraphrase for IDIOMS, out of a total of 100 sentences,",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/sarcasm-hyperbole-humor-paraphrase for IDIOMS, out of a total of 100 sentences, 14 sentences were modified by the transformer 