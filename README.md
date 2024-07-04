# Хакатон МФТИ 2024

Веб-сервис: http://whereismyplace.itatmisis.ru:8501/
Презентация: [тык](https://drive.google.com/file/d/1lyFPwIg8UI6X0OlbvL3-pieV4R_ENZq0/view?usp=sharing)

## Кейс "Мультикамерное распознавание места"

> На хакатоне Вам необходимо разработать метод распознавания места на территории кампуса МФТИ для мобильного робота Clearpath Husky (см. рисунок 1) по синхронизированным данным его двух бортовых камер, дополнительно можно использовать семантическую информацию о сцене в виде масок семантической сегментации изображений и результаты детекции и распознавания текстовых надписей. Такая постановка задачи является современным трендом в искусственном интеллекте и Вам предоставляется познакомиться с её существующими решениями и предложить свое.

## Предложенное решение

- использовали `mink_quantization_size=0.06` в ITLPCampusOutdoor, это дало прирост к качеству (также пробовали 0.01, 0.05, 0.08 и классические 0.5)
- в качестве модели для извелечения фичей из изображений использовали `ConvNeXtTinyFeatureExtractor`, учили 7 эпох (еще тестировали ResNet18 и ResNet50, они показали результаты хуже, пробовали разное количество эпох в диапазоне от 6 до 16)
- шедулер `cosine_schedule_with_warmup` (WARMUP_STEPS = 90, T_MAX = 1800), оптимизатор `AdamW` (WEIGHT_DECAY = 0.003), функция потерь `BatchHardTripletMarginLoss` (margin=0.5, также пробовали другие значения, однако они показывали худший результат)
- `GeM` pooling, `Concat` fusion вместо `Add`
- для быстрого поиска заменили KDTree на `faiss`
- визуализировали с помощью `matplotlib` и `streamlit`, есть возможность выбрать любую точку и посмотреть на мини-карте местности истинное положение и предсказанное, увидеть дистанцию между ними, а также изображения с обеих камер с двух этих мест

## Наработки

- сначала попробовали использовать семантические маски напрямую, это только ухудшило скор, поэтому был написана функция для удаления из масок динамических объектов (люди, машины, животные), однако нормально встроить это в класс датасета не получилось, хотя мы долго и упорно старались :(
- для получения эмбеддингов текстов использовали `e5-multilingual-small` и добавили эмбеддинги текстов с двух камер в датасет:

```
{'idx': tensor(600),
 'pose': tensor([-2.2312e+02,  5.5457e+00, -2.3279e+00,  7.6497e-03,  3.0006e-02,
          3.3897e-01,  9.4029e-01]),
 'image_front_cam': tensor([[[ 1.0844,  1.0844,  1.0673,  ...,  1.1529,  1.1358,  1.1358],
          [ 1.0844,  1.0844,  1.0673,  ...,  1.1529,  1.1358,  1.1358],
          [ 1.0844,  1.0844,  1.0844,  ...,  1.1529,  1.1358,  1.1358],
          ...,
),
 'text_front_cam': tensor([[ 0.1494, -0.0085, -0.1771, -0.5213,  0.4211, -0.0120, -0.0325,  0.1309,
           0.4322,  0.1151,  0.3700, -0.0126,  0.2740, -0.2194, -0.1466,  0.1061,
           0.3116, -0.2171, -0.1378, -0.1360,  0.2029, -0.1187, -0.3346,  0.0911,
           ...,
],
        grad_fn=<DivBackward0>),
 'image_back_cam': tensor([[[-1.8610, -1.8439, -1.8268,  ...,  1.1700,  1.1872,  1.2043],
          [-1.6727, -1.7240, -1.8097,  ...,  1.1529,  1.1700,  1.2385],
          [-1.6898, -1.7754, -1.8953,  ...,  1.1358,  1.1529,  1.2043],
          ...,
),
 'text_back_cam': tensor([[ 3.0309e-01, -4.5260e-02, -1.7883e-01, -4.4682e-01,  3.6245e-01,
          -1.4150e-01,  1.2442e-01,  2.3768e-01,  2.6231e-01,  5.4371e-02,
           2.7937e-01,  2.3813e-02,  2.8877e-01, -4.9143e-03, -1.0536e-01,
           ...,
],
        grad_fn=<DivBackward0>)}
```

- однако, если использовать concat fusion для эмбеддингов изображений и текстов, то мы не влезем в память с таким решением. чтобы использовать add, можно было попробовать clip эмбеддинги, однако на всех лидербордах clip показывал довольно плохие результаты относительно других моделей ([лидерборд моделей](https://paperswithcode.com/sota/visual-place-recognition-on-17-places)), поэтому от этого мы тоже отказались
- попытались в LateFusionModel сделать attention fusion, однако ничего хорошего тоже не получили :(
- попробовали реализовать логику реранжирования из статьи "Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition", однако это только ухудшило качество

## Содержание репозитория
- `notebooks` - здесь лежат наши попытки поработать с текстами и масками, реранжирование, faiss и итоговый код
  - [mipt_convnext.ipynb](https://nbviewer.org/github/l1ghtsource/mipt-hackathon/blob/main/notebooks/mipt_convnext.ipynb)
  - [text_masks_faiss_solution.ipynb](https://nbviewer.org/github/l1ghtsource/mipt-hackathon/blob/main/notebooks/text_masks_faiss_solution.ipynb)
  - []()
- `src` - а здесь фронт и бэк для нашего творения
