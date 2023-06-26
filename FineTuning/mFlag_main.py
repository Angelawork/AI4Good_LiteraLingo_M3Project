import transformers
import torch

print(transformers.__version__)
print(torch.__version__)


from mFLAG.model import MultiFigurativeGeneration
from mFLAG.tokenization_mflag import MFlagTokenizerFast
tokenizer = MFlagTokenizerFast.from_pretrained('laihuiyuan/mFLAG')
model = MultiFigurativeGeneration.from_pretrained('laihuiyuan/mFLAG')

sentence="The baby would be my baby to teach."
type="hyperbole"
# a token (<hyperbole>) is added at the beginning of the source sentence to indicate its figure of speech
inp_ids = tokenizer.encode("<hyperbole> That joke is so old, the last time I heard it I was riding on a dinosaur.", return_tensors="pt")
# the target figurative form
fig_ids = tokenizer.encode("<literal>", add_special_tokens=False, return_tensors="pt")
outs = model.generate(input_ids=inp_ids[:, 1:], fig_ids=fig_ids, forced_bos_token_id=fig_ids.item(), num_beams=5, max_length=60,)
text = tokenizer.decode(outs[0, 2:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
# special tokens: <literal>, <hyperbole>, <idiom>, <sarcasm>, <metaphor>, or <simile>
print(text)