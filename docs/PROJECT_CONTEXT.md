# Project Context — читать в начале новой сессии

## Что строим

Небольшое AI web-приложение для общения людей, использующих азербайджанский жестовый язык.

Название рабочего проекта: `sign-language-ai`.

Целевая аудитория: Azerbaijan / Azerbaijani-speaking users.

## Почему AzSLD

Мы сознательно НЕ начинаем с ASL. Нужен именно Azerbaijani Sign Language.

Используем AzSLD — Azerbaijani Sign Language Dataset.

Основной первый набор:
- `AzSLD_Words_100.zip`
- размер примерно 1.1 GB
- 100 классов слов

Позже:
- `AzSLD_Words_200.zip`
- `AzSLD_Fingerspelling.zip`
- `AzSLD_Sentences.zip`

## Что уже работает

### Frontend
HTML/CSS/JS.

### Camera
`navigator.mediaDevices.getUserMedia()`.

### MediaPipe
`@mediapipe/tasks-vision`
`HandLandmarker`
`runningMode: VIDEO`
до 2 рук.

MediaPipe даёт landmarks, но НЕ переводит жесты в слова.

## Что пока НЕ работает

Реальный классификатор AzSL.

Сейчас интерфейс показывает примерно:

`Hand detected`

Нужно заменить это на:

`Salam`

`Sağ ol`

и т.д.

## Архитектурное решение

Realtime inference желательно выполнять в браузере:

```text
User Browser
 ├─ Camera
 ├─ MediaPipe
 └─ AzSL model
       ↓
   Azerbaijani text

Node/Express
 ├─ serving app
 └─ future API
```

Причина: не отправлять каждый видеокадр на сервер, уменьшить latency и стоимость.

## Training

Для обучения допускается Python.

План:

```text
AzSLD videos
  ↓
MediaPipe landmark extraction
  ↓
normalized sequences
  ↓
temporal model
  ↓
100-class classifier
  ↓
ONNX / browser-compatible model
  ↓
frontend
```

Кандидаты модели:
- GRU/LSTM — первый простой вариант
- Transformer — если нужна более высокая точность/длинные последовательности

## Deployment

Render подключён к GitHub.

Для Node Web Service:
- Build: `npm install`
- Start: `npm start`
- server должен слушать `0.0.0.0`
- порт: `process.env.PORT`

Render URL будет `*.onrender.com`.

## Не забыть

Если меняем realtime-код:
- не делать `console.log` на каждом frame
- не печатать landmarks в консоль
- не обновлять DOM на каждом frame без необходимости
- желательно throttling UI updates

## Как продолжить

В новой сессии сначала прочитать этот файл и `ROADMAP.md`.

Фраза для новой сессии:

> Продолжаем Sign Language AI. Прочитай docs/PROJECT_CONTEXT.md и docs/ROADMAP.md. AzSLD_Words_100.zip уже скачан. MediaPipe camera работает. Следующий этап — подготовить dataset и обучить первую 100-классовую AzSL модель.
