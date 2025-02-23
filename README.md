
<h1 align="center">Military Message Classifier</h1> 

<p align="center"> 
  <strong>Розширена веб-додаток для автоматизованого аналізу військових повідомлень із Telegram-каналів.</strong> 
</p> 

<p align="center"> 
  <a href="https://github.com/yourusername/military-message-classifier/actions/workflows/ci.yml"> 
  </a> 
  <a href="https://github.com/yourusername/military-message-classifier/blob/main/LICENSE"> 
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Ліцензія: MIT"> 
  </a> 
  <a href="https://www.python.org/downloads/release/python-3100/"> 
    <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"> 
  </a> 
  <a href="https://angular.io/"> 
    <img src="https://img.shields.io/badge/angular-14.0.0-red.svg" alt="Angular 14.0.0"> 
  </a> 
  <img src="https://img.shields.io/badge/status-in%20development-orange.svg" alt="Статус: В розробці">
</p> 

<p align="center"> 
  <strong>Military Message Classifier</strong> автоматизує збір, обробку та класифікацію повідомлень із Telegram-каналів. Використовуючи NLP і машинне навчання, він витягує важливі військові дані, розпізнає об’єкти, такі як зброя та локації, і класифікує повідомлення за критеріями військового досвіду. Підходить для військового аналізу, надаючи корисну розвідувальну інформацію через автоматизовані звіти. 
</p>

---
## Зміст
- [Функціонал](#функціонал)
- [Архітектура](#архітектура)
- [Встановлення](#встановлення)
- [Використання](#використання)


---
## Функціонал
- **Збір повідомлень**: Автоматично отримує повідомлення із зазначених Telegram-каналів.
- **Попередня обробка тексту**: Очищає та готує текст повідомлень до аналізу.
- **Розпізнавання сутностей**: Виявляє іменовані сутності (особи, локації, зброя) за допомогою NLP.
- **Класифікація досвіду**: Визначає військовий досвід у повідомленнях (моделі BERT, XGBoost).
- **Дедуплікація**: Видаляє дублікати повідомлень для забезпечення унікальності даних.
- **Генерація звітів**: Створює звіти ODCR (Спостереження, Обговорення, Висновки, Рекомендації).
- **Веб-інтерфейс**: Дозволяє переглядати, фільтрувати та взаємодіяти з класифікованими повідомленнями.

---
## Архітектура
Проєкт поділений на бекенд і фронтенд:

### **Бекенд** (Flask)
Відповідає за взаємодію з базою даних, обробку повідомлень і API:
- `database/` – Скрипти ініціалізації БД (`init_db.py`).
- `experience/` – Класифікація військового досвіду (`exp.py`).
- `services/` – NLP-модулі:
  - `ner.py` – Розпізнавання іменованих сутностей.
  - `odsr.py` – Генерація звітів ODCR.
  - `preproc.py` – Попередня обробка тексту.
  - `remove_duplicates.py` – Дедуплікація.
- `utils/` – Допоміжні файли (`weapon.txt`).
- `main.py` – Основний скрипт Flask-додатку.

### **Фронтенд** (Angular)
Забезпечує веб-інтерфейс:
- `app.component.*` – Основний компонент додатку.
- `app.config.ts`, `app.routes.ts` – Конфігурація та маршрутизація.

### **Моделі** (ML/NLP)
Скрипти машинного навчання:
- `Sberbank_AI_ruBERT.py` – Класифікація на базі BERT.
- `XGBoost_tokenizer_BERT.py` – XGBoost із вбудовуваннями BERT.
- `use_model.py` – Скрипт використання моделей.

### **Попередня обробка** (Jupyter)
Jupyter-зошити для обробки та маркування даних:
- `get_new_data.ipynb` – Отримання та обробка даних.
- `label_data.ipynb` – Маркування даних.
- `levenshtein.ipynb` – Дедуплікація за допомогою відстані Левенштейна.

![image](https://github.com/user-attachments/assets/5b946e2b-3a6e-4f83-a01d-9d1109a9e3d8)
---
## Встановлення

### **Клонування репозиторію**
```bash
git clone https://github.com/yourusername/military-message-classifier.git
cd military-message-classifier
```

### **Налаштування бекенду**
Встановлення залежностей Python:
```bash
pip install -r requirements.txt
```

Ініціалізація бази даних:
```bash
python backend/database/init_db.py
```

### **Налаштування фронтенду**
Перехід до директорії фронтенду:
```bash
cd frontend
```

Встановлення залежностей:
```bash
npm install
```

### **Налаштування змінних оточення**
Вкажіть API-ключі (Telegram API, OpenAI API) та інші конфігураційні змінні.

---
## Використання

### **Запуск бекенд-сервера**
```bash
python backend/main.py
```

### **Запуск фронтенд-сервера**
```bash
cd frontend
ng serve
```

### **Доступ до додатку**
Відкрийте браузер та перейдіть на: [http://localhost:4200](http://localhost:4200)

### **Функціональність**
- Отримання повідомлень із Telegram-каналів.
- Фільтрація та взаємодія з класифікованими повідомленнями.
- Генерація та завантаження звітів ODCR.

---

