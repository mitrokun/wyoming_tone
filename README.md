# Wyoming t-one

Тестовая (с неоднозначными решениями) реализация сервера для модели от желтого банка.


```
# 1. Клонирование репозитория
git clone https://github.com/mitrokun/wyoming_tone.git
cd wyoming_tone
# делаем скрипт запуска исполняемым
cd tone_asr
chmod +x run
cd ..

# 2. Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# 3. Установка всех необходимых Python-библиотек
pip install resampy wyoming
pip install git+https://github.com/voicekit-team/T-one.git

# 4. Запуск сервера
# возвращаемся в домашний каталог
cd
# абсолютный путь должен работать из любого места.
# можно добавлять в хрон или systemd
 ~/wyoming_tone/tone_asr/run --uri tcp://0.0.0.0:10303
```
Для мощных: `--decoder beam_search` скачивает и включает 5гб+ LM, что улучшает результат.
