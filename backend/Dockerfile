FROM python:3.10.16-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-dotenv
RUN pip install pymorphy3
RUN pip install pymorphy3-dicts-uk
RUN python -m spacy download ru_core_news_md

COPY backend/ .

RUN python database/init_db.py

EXPOSE 5001

CMD ["python", "main.py", "--host", "0.0.0.0"]

