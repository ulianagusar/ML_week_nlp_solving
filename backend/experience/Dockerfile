FROM python:3.10.16-slim

WORKDIR /app/backend/experience

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5003

CMD ["python", "exp.py", "--host", "0.0.0.0"]



#  FROM python:3.10.16-slim
 
#  WORKDIR /app/backend/experience
 
#  COPY requirements2.txt /app/
 
#  RUN pip install --no-cache-dir -r requirements.txt
 
#  COPY backend/ /app/
 
#  EXPOSE 5003
 
#  CMD ["python", "experience/exp.py"]
