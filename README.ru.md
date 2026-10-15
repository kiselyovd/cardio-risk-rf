# Классификация наличия CVD

[Полный README / English](README.md) · [Модели](https://huggingface.co/kiselyovd/cardio-risk-rf) · [Документация](https://kiselyovd.github.io/cardio-risk-rf/)

Опубликованные LightGBM и RandomForest используют 11 признаков Sulianova и целевую метку `cardio`: наличие заболевания на момент обследования. Это отдельная задача от исторического Framingham/TenYearCHD.

Оригинальный CSV найден и проверен хешем; оба model bundle повторно оценены на 10 501 test-строке. [Таблица результатов](README.md#current-artifact-and-evaluation) генерируется из machine-readable manifest. [Полный replay](reports/replay-native.json) сохраняет модели, данные, split и массивы предсказаний.

Возраст на входе модели — округлённые годы. Сохранён исходный split 48 999/10 500/10 501: это результат float/ceil алгоритма, совпадающий с числом training-строк в RF. Не меняем его незаметно при проверке старых весов.

Канонические joblib-файлы содержат dict bundle; estimator лежит в `bundle["model"]`. Отдельный plain-Pipeline алиас для HF-клиентов создаётся из этих же весов и проверяется на равенство предсказаний.

[Команды запуска и повторения оценки](README.md) · [Контракт и единицы](docs/EVALUATION.md) · [Лицензии](docs/LICENSES.md). Код MIT; лицензия исходных данных у издателя отмечена как Unknown. Клиническая валидация не проводилась.
