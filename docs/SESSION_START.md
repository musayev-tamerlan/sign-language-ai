# SESSION START — QUICK CONTEXT

Если новая сессия началась, прочитай:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/ROADMAP.md`
3. `docs/ARCHITECTURE.md`

## One-sentence context

Мы делаем realtime веб-приложение для распознавания **азербайджанского жестового языка** через камеру, начиная с 100 слов AzSLD.

## Current state

- Node + Express: есть
- Camera: работает
- MediaPipe Hand Landmarker: работает
- Render: аккаунт создан
- GitHub: подключён к Render
- AzSLD_Words_100.zip: скачан локально
- Model training: ещё не сделан
- Production deployment: ещё не завершён

## NEXT TASK

**Не начинать заново.**

Следующая задача:

```text
1. Распаковать AzSLD_Words_100.zip
2. Посмотреть реальную структуру
3. Определить labels/classes
4. Сделать Python dataset loader
5. Extract MediaPipe landmarks
6. Сделать train/validation/test split
7. Обучить первую GRU/LSTM модель на 100 словах
8. Оценить accuracy/F1
9. Экспортировать модель
10. Подключить модель к browser
```

## Important

Не скачивать пока `AzSLD_Sentences.zip`.

Не пытаться распознавать весь язык сразу.

Сначала добиться стабильного MVP на 100 словах.

## Prompt for the next ChatGPT session

```text
Продолжаем проект Sign Language AI.

Прочитай:
- docs/PROJECT_CONTEXT.md
- docs/ROADMAP.md
- docs/ARCHITECTURE.md
- docs/SESSION_START.md

Мы делаем Azerbaijani Sign Language recognition.

Node.js + Express + browser camera + MediaPipe уже работают.
Render подключён к GitHub.
AzSLD_Words_100.zip (~1.1 GB) уже скачан и находится локально.

Не начинай проект заново.

Продолжай со следующего незавершённого шага ROADMAP.
Сейчас нужно изучить структуру AzSLD_Words_100 и подготовить training pipeline.
```
