FROM python:3.10.16-slim

WORKDIR /app/backend

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-dotenv pymorphy3 pymorphy3-dicts-uk
RUN python -m spacy download ru_core_news_md

COPY . .

RUN python database/init_db.py

EXPOSE 5001

CMD ["python", "main.py", "--host", "0.0.0.0"]


# FROM python:3.10.16-slim

#  WORKDIR /app
 
#  COPY requirements2.txt .
 
#  RUN pip install --no-cache-dir -r requirements.txt
#  RUN pip install python-dotenv
#  RUN pip install pymorphy3
#  RUN pip install pymorphy3-dicts-uk
#  RUN pip install python-dotenv pymorphy3 pymorphy3-dicts-uk
#  RUN python -m spacy download ru_core_news_md
 
#  COPY backend/ .
 
#  RUN python database/init_db.py
 
#  EXPOSE 5001
 
#  CMD ["python", "main.py", "--host", "0.0.0.0"]
