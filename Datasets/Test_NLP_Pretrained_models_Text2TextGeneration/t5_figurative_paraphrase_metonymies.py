#Metonymy is a figure of speech
#From Wikipedia:
#Metonymy is similar to synecdoche, but instead of a part representing the whole, a related object or part of a related 
#object is used to represent the whole.[2] Often it is used to represent the whole of an abstract idea. 
#Example: The phrase "The king's guns were aimed at the enemy," using 'guns' to represent infantry.
#Example: The word 'crown' may be used metonymically to refer to the king or queen, and at times to the law of the land.

#Description: The methaphor.csv file has 2 columns:metonymic sentence and literal sentence
#Sources:
#https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads&search=figurative-nlp
#1.figurative-nlp/t5-figurative-paraphras:https://huggingface.co/figurative-nlp/t5-figurative-paraphrase?text=No%2C+I%27d+say+this+was+dumbest
#This model can convert the figurative/metaphorical expression to the literal expression.
#2.figurative-nlp/sarcasm-hyperbole-humor-paraphrase:https://huggingface.co/figurative-nlp/sarcasm-hyperbole-humor-paraphrase 
#There is no description about this pretained model in Hugging face
#3.NLP(text2TextGeneration)


import pandas as pd
metonymic_df = pd.read_csv(r"C:\Users\adela\Workspace-Python\t5-figurative-paraphrase\Metonymies.csv")
metonymic_df.loc[0].index
#print(metonymic_df.loc[0].index,"\n")
metonymic_df.Metonymic_Sentence.index
#print(metonymic_df.Metonymic_Sentence.index,"\n") 
metonymic_df.Literal_Sentence.index
#print(metonymic_df.literal_expression.index,"\n")
metonymic_column = metonymic_df["Metonymic_Sentence"]
#print(metonymic_column,"\n")
#Iterating over column
list_metonymic_column = list(metonymic_column)[:100]
#list_metonymic_column = metonymic_column.values

#print(list_metonymic_column[:3])
#for i in list_metonymic_column[:10]:
    #print(i)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#There are 2 models : figurative-nlp/t5-figurative-paraphrase and figurative-nlp/sarcasm-hyperbole-humor-paraphrase
model_name = "figurative-nlp/t5-figurative-paraphrase"
#model_name = "figurative-nlp/sarcasm-hyperbole-humor-paraphrase"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)



input_ids = tokenizer(list_metonymic_column, padding=True,
    truncation=False,
    max_length=512,return_tensors="pt").input_ids
print(input_ids)
outputs = model.generate(input_ids)
print("OUT: ", outputs)


counter_int = 0
for i in range(len(outputs)):
    result = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(i, "TRANSFORMER: ", result)
    print(i, "METONYMIES IN LANGUAGE: ", list_metonymic_column[i],"\n")
    if result != list_metonymic_column[i]:
        counter_int +=1

print("figurative-nlp/t5-figurative-paraphrase model for METONYMIES, out of a total of 100 sentences,",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/t5-figurative-paraphrase model for METONYMIES, out of a total of 100 sentences, 100 sentences were modified by the transformer
#print("figurative-nlp/sarcasm-hyperbole-humor-paraphrase model for METONYMIES, out of a total of 100 sentences,",counter_int,"sentences were modified by the transformer")
#Get:figurative-nlp/sarcasm-hyperbole-humor-paraphrase model for METONYMIES, out of a total of 100 sentences, 98 sentences were modified by the transformer