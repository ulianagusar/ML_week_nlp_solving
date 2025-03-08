from telethon.sync import TelegramClient

api_id = 24103680
  # Ваш API ID
api_hash = "9751db4cb8fe8a9d099e862e6316cbe5"  # Ваш API Hash

client = TelegramClient("military_bot", api_id, api_hash)
client.start()  # Telegram попросить ввести код, який прийде у ваш обліковий запис
