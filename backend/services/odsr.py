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




import openai
import json
openai.api_key = api_key


def generate_odcr_report(input_message):
    """Generate ODCR report with error handling"""
    try:
        prompt = f"""
        Convert the following message into an ODCR report and return the result strictly in valid JSON format.

        Message:
        {input_message}

        The JSON structure should be:
        {{
            "O": "Observation - Briefly describe the issue or problem and its resolution.",
            "D": "Discussion - Expand on the observation with key details (who, what, where, when, why, how) and its impact on operations.",
            "C": "Conclusion - Summarize key points and support the recommendation.",
            "R": "Recommendation - Suggest actions to resolve the issue, including responsible parties.",
            "T": "Type - Specify which branch of the military this information may be useful for: Наземні війська, Повітряні війська, Морські війська, Десантно-штурмові війська, Війська підтримки, or None."
        }}

        Answer only in Russian and return strictly valid JSON with no additional text.
        """

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        # Отримуємо текст відповіді
        content = completion['choices'][0]['message']['content'].strip()
        
        # Перетворюємо JSON-рядок у Python-об'єкт
        report = json.loads(content)
        
        return report.get("O", ""), report.get("D", ""), report.get("C", ""), report.get("R", ""), report.get("T", "") 

    except Exception as e:
        print(f"Error generating ODCR report: {e}")
        return "", "", "", "", ""


# text = '''Во время учебного полета была потеряна связь с самолетом на расстоянии 1,7 км. Последние показания по пичу дали гипотезу о возможной неисправности сервы.
# Мы пытались найти его на некотором расстоянии от последних координат, в сторону ветра, предполагая, что самолет мог унести по ветру. В итоге зашли в радиус 25 м от точки координат последней зафиксированной.
# Ожидалось, что самолет еще некоторое время летел и был унесен в сторону ветра.
# Таким образом, при потере самолета следует начинать поиски с самого простого и/или ближайшего варианта. По возможности иметь резервную (аварийную) автономную систему передачи координат самолета.
# Предлагаю добавить в самолет аварийную систему GPS для определения координат местоположения борта.'''


# o, d, c, r, t = generate_odcr_report(text)

# print (o, '\n', d, '\n', c, '\n', r, '\n', t)




def get_o(mes):

    return "o"

def get_d(mes):

    return "d"

def get_c(mes):

    return "s"


def get_r(mes):

    return "r"

# def generate_odcr_report(input_message):
#     return "", "", "", "", ""