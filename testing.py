import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)


def greedy_decoding (inp_ids,attn_mask):
  greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
  Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return Question.strip().capitalize()


def beam_search_decoding (inp_ids,attn_mask):
  beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
  return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding (inp_ids,attn_mask):
  topkp_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               do_sample=True,
                               top_k=40,
                               top_p=0.80,
                               num_return_sequences=3,
                                no_repeat_ngram_size=2,
                                early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]
  return [Question.strip().capitalize() for Question in Questions]


#creating streamlit app

import streamlit as st
import time

st.title("Boolean Question Generation")

st.write('Assignment for AI Planet')

st.write('Enter the context and click on Generate Questions to get the Boolean Questions')

passage = st.text_area("Enter the context here")

#Question generation
truefalse = "yes"
text = "truefalse: %s passage: %s </s>" % (passage, truefalse)

if st.button('Generate Questions'):
    #roller
    with st.spinner('Generating Questions...'):
        time.sleep(5)
    st.success('Done!')

    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = beam_search_decoding(input_ids,attention_masks)
    for out in output:
        #display the statement generated along with the option True/ False.  
        st.radio(out,('True','False'))







