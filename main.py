import os
import time
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes

st.title("Let's Translate")


# Load models
model_name_dict = {'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M'}
model_dict = {}

for call_name, real_name in model_name_dict.items():
    st.write(f'Loading model: {call_name}')
    model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
    tokenizer = AutoTokenizer.from_pretrained(real_name)
    model_dict[call_name + '_model'] = model
    model_dict[call_name + '_tokenizer'] = tokenizer


# Define translation function
def translation(source, target, text):
    start_time = time.time()
    source = flores_codes['English']  # Set source language to English
    target = flores_codes[target]

    model = model_dict['nllb-distilled-600M_model']
    tokenizer = model_dict['nllb-distilled-600M_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target)
    output = translator(text, max_length=1000)

    end_time = time.time()

    output = output[0]['translation_text']
    result = {
        'inference_time': end_time - start_time,
        'source': source,
        'target': target,
        'result': output
    }
    return result

# Streamlit UI
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv","xlsx"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # List of target languages
    target_languages = ['Italian', 'Spanish', 'German', 'French', 'Russian','Modern Standard Arabic','Urdu', 'Marathi', 'Tamil', 'Hindi']

    # Add a "Submit" button
    if st.button("Submit"):
        # Translate for each target language and add a column for each translation
        for lang in target_languages:
            translations = []
            for question in df['Question']:
                result = translation('English', lang, question)
                translations.append(result['result'])
            st.write(f"translation is done for {lang}")
            df[lang] = translations

        # Display the translated DataFrame with added columns
        st.write("Translated DataFrame:")
        st.dataframe(df)

        # Provide a download link for the result DataFrame as a CSV file
        result_file_name = "translated_results.csv"
        df.to_csv(result_file_name, index=False)
        st.markdown(
            f"Download the translated results as a CSV file: [Download {result_file_name}](./{result_file_name})"
        )
