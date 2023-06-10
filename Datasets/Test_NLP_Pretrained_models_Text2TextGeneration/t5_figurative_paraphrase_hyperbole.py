#Hyperbola-Hyperbole is a figure of speech that uses an exaggerated or extravagant statement to create a strong emotional response.

#Description: The hyperbole.csv file has 2 columns: hyperboles and literal expression of the hyperboles
#Sources:
#https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=figurative-nlp
#1.figurative-nlp/t5-figurative-paraphras:https://huggingface.co/figurative-nlp/t5-figurative-paraphrase?text=No%2C+I%27d+say+this+was+dumbest
#2.figurative-nlp/sarcasm-hyperbole-humor-paraphrase:https://huggingface.co/figurative-nlp/sarcasm-hyperbole-humor-paraphrase 
#3.NLP(text2TextGeneration)

import pandas as pd
hyperbole_df = pd.read_csv(r"C:\Users\adela\Workspace-Python\t5-figurative-paraphrase\hyperbole.csv")
hyperbole_df.loc[0].index
# print(hyperbole_df.loc[0].index,"\n")#Index(['literal meaning', 'hyperbole'], dtype='object')
hyperbole_df.hyperbole.index
# print(hyperbole_df.hyperbole.index,"\n")#RangeIndex(start=0, stop=102886, step=1) 
hyperbole_df.literal_expression.index
# print(hyperbole_df.literal_expression.index,"\n")#RangeIndex(start=0, stop=102886, step=1) 
hyperbole_column = hyperbole_df["hyperbole"]
# print(hyperbole_column,"\n")#Name: hyperbole, Length: 102886, dtype: object

#Iterating over column
list_hyperbole_column = list(hyperbole_column)[:100]
# list_hyperbole_column = hyperbole_column.values

# print(list_hyperbole_column[:3])
# for i in list_hyperbole_column[:10]:
#     print(i)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#There are 2 models:figurative-nlp/t5-figurative-paraphrase and figurative-nlp/sarcasm-hyperbole-humor-paraphrase
model_name = "figurative-nlp/t5-figurative-paraphrase"
#model_name = "figurative-nlp/sarcasm-hyperbole-humor-paraphrase"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)



input_ids = tokenizer(list_hyperbole_column, padding=True,
    truncation=False,
    max_length=512,return_tensors="pt").input_ids
print(input_ids)
outputs = model.generate(input_ids)
print("OUT: ", outputs)


counter_int = 0
for i in range(len(outputs)):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(i, "TRANSFORMER: ", result)
    print(i, "HYPERBOLIC LANGUAGE: ", list_hyperbole_column[i],"\n")
    if result != list_hyperbole_column[i]:
        counter_int +=1

#print("figurative-nlp/sarcasm-hyperbole-humor-paraphrase for HYPERBOLE, out of a total of 100 sentences,",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/sarcasm-hyperbole-humor-paraphrase for HYPERBOLE, out of a total of 100 sentences, 50 sentences were modified by the transformer
print("figurative-nlp/t5-figurative-paraphrase model for HYPERBOLE, out of a total of 100 sentences,",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/t5-figurative-paraphrase model for HYPERBOLE, out of a total of 100 sentences, 21 sentences were modified by the transformer

    
   