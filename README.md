# Sign Language AI — Azərbaycan İşarət Dili

AI-приложение для распознавания **азербайджанского жестового языка (AzSL / Azərbaycan İşarət Dili)** через веб-камеру и перевода жестов в азербайджанский текст.

## Dataset attribution

This project uses and includes derivatives of the **AzSLD — Azerbaijani Sign Language Dataset** by N. Alishzade and J. Hasanov (2025), DOI: [10.5281/zenodo.14222948](https://doi.org/10.5281/zenodo.14222948), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Текущий статус

- Node.js + Express: готово
- Browser webcam: работает
- MediaPipe Hand Landmarker: работает
- Получение landmarks руки: работает
- Лишние `console.log` из realtime-цикла: убрать/не использовать
- Render account: создан
- GitHub подключён к Render
- `AzSLD_Words_100.zip` (~1.1 GB): скачан и находится в локальной папке проекта
- Реальная модель распознавания AzSL ещё НЕ обучена
- Production deployment ещё НЕ завершён

## Главная цель

```text
Camera
  ↓
MediaPipe
  ↓
Hand landmarks / video sequence
  ↓
AzSL recognition model
  ↓
Azərbaycan sözü
  ↓
Sentence buffer
  ↓
Azərbaycan dilində mətn
```

Первый MVP: **100 слов AzSLD**.

После этого:
1. fingerspelling / dактиль
2. 200 слов
3. continuous sentence recognition
4. нормализация/построение естественных предложений
5. text-to-speech
6. production deployment

## Важно

Не загружать `AzSLD_Words_100.zip` в GitHub.

Не скачивать пока `AzSLD_Sentences.zip` (43.8 GB). Он понадобится позже для continuous sentence recognition.

## ML pipeline

Текущее состояние и безопасная проверка данных: [docs/ML_STATUS.md](docs/ML_STATUS.md).

## Веб-интерфейс и Render

Интерфейс на азербайджанском. Запуск: `npm ci`, `npm run build`, `npm start`. Настройки публикации и проверки: [docs/RENDER.md](docs/RENDER.md).

## Улучшение модели

Аудит слабых слов и безопасная запись новых размеченных примеров: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md).
