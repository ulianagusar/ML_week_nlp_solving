# !pip install transformers sentencepiece
# !pip install transformers torch


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, MBartTokenizer, MBartForConditionalGeneration

# def gen_o(text: str, model_name: str = "cointegrated/rut5-base-absum"):
#     """Генерация спостереження (Observation)"""
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
#     prompt = "У чем проблема: " + text
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
#     output_ids = model.generate(input_ids, num_beams=5, early_stopping=True, max_length=256)
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# def gen_d(text, max_length=512, min_length=50, lang_code="ru_RU"):
#     """Генерация обоснования (Discussion)"""
#     model_name = "facebook/mbart-large-50"
#     tokenizer = MBartTokenizer.from_pretrained(model_name)
#     model = MBartForConditionalGeneration.from_pretrained(model_name)

#     # Переконаємося, що lang_code існує
#     if lang_code not in tokenizer.lang_code_to_id:
#         print(f"Warning: lang_code {lang_code} not found! Using 'ru'.")
#         lang_code = "ru"

#     prompt = "Объясните, почему это произошло: " + text  # Повернув промпт!
#     inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
#     output_ids = model.generate(
#         inputs, max_length=max_length, min_length=min_length, length_penalty=2.0,
#         num_beams=4, early_stopping=True, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code]
#     )
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# def gen_c(text: str, model_name: str = "IlyaGusev/rut5_base_sum_gazeta"):
#     """Генерация вывода (Conclusion)"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     prompt = "Сформулируйте краткий и четкий вывод на основе текста: " + text
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
#     output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# def gen_r(text: str, model_name: str = "cointegrated/rut5-base-absum"):
#     """Генерация рекомендации (Recommendation)"""
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
#     prompt = "Опишите, что произошло и что нужно делать: " + text
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
#     output_ids = model.generate(input_ids, num_beams=5, early_stopping=True, max_length=256)
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)




def get_o(mes):

    return "o"

def get_d(mes):

    return "d"

def get_c(mes):

    return "s"


def get_r(mes):

    return "r"

