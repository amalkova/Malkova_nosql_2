# Semantic Search for arXiv Papers

Домашнє завдання з NoSQL, присвячене векторному пошуку, ембеддингам, Pinecone, chunking, BM25 та hybrid search через Reciprocal Rank Fusion.

У роботі побудовано пошуковий рушій для наукових статей arXiv, який дозволяє шукати документи не лише за точним збігом слів, а й за семантичною близькістю текстів.

Для експериментів використано підмножину з 10 000 статей arXiv.

---

# Структура проєкту

```text
.
├── .env
├── .gitignore
├── requirements.txt
├── data/
│   └── arxiv_subset.parquet
├── embeddings/
│   └── embeddings.npy
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
└── README.md
```

Файли `.env`, `data/` та `embeddings/` не додаються до Git-репозиторію.

---

# Налаштування оточення

## Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Встановлення залежностей

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
pinecone-client==4.1.0
sentence-transformers==3.0.1
rank-bm25==0.2.2
kaggle==1.6.14
pandas==3.0.3
numpy==2.4.6
tqdm==4.66.4
pyarrow==23.0.1
python-dotenv==1.0.1
```

## Pinecone API key

У корені проєкту створюється файл `.env`:

```text
PINECONE_API_KEY=<your-api-key>
```

`.env` додано до `.gitignore`.

---

# Частина 1 — Підготовка даних та ембеддингів

## 1.1 Завантаження arXiv Dataset

Dataset було завантажено через Kaggle:

```bash
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
```

Початковий файл:

```text
arxiv-metadata-oai-snapshot.json
```

має формат JSONL та займає приблизно 5.2 GB.

Скрипт:

```text
scripts/01_prepare_data.py
```

читає JSONL построково, не завантажуючи весь файл у пам'ять.

До підмножини включаються записи, які мають `title` та `abstract`.

Для кожної статті зберігаються:

```text
id
title
abstract
authors
year
category
```

Результат:

```text
data/arxiv_subset.parquet
```

Фактичний результат:

```text
rows: 10000
```

Приклад першого запису:

```text
id: 0704.0001
title: Calculation of prompt diphoton production cross sections at Tevatron and LHC energies
year: 2007
category: hep-ph
```

---

# 1.2 Вибір інструментів

## Pinecone vs Qdrant vs Chroma

| Система | Deployment | Ліцензія / модель | Коли я б обрала |
|---|---|---|---|
| Pinecone | Fully managed cloud / serverless | Комерційний managed service | Коли не хочу адмініструвати інфраструктуру та потрібен готовий production-oriented vector search |
| Qdrant | Self-hosted, Qdrant Cloud, Private/Hybrid Cloud | Apache 2.0 | Коли потрібен контроль над інфраструктурою, можливість self-hosting та масштабований production search |
| Chroma | Local, Docker, Chroma Cloud | Apache 2.0 | Для прототипів, локальної розробки, невеликих RAG-проєктів або швидкого старту |

### Pinecone

Pinecone зручний як managed solution: не потрібно самостійно управляти HNSW-індексами, репліками, storage або кластером.

Для цього домашнього завдання це зручно, тому що достатньо створити index через API та завантажити vectors.

Недолік — залежність від зовнішнього SaaS та лімітів тарифного плану.

### Qdrant

Qdrant є open-source vector database з Apache 2.0 license.

Його можна запускати локально через Docker, на власній інфраструктурі або використовувати Qdrant Cloud.

Це хороший варіант, коли потрібні:

- self-hosting;
- контроль над ресурсами;
- metadata filtering;
- distributed deployment;
- production vector search.

### Chroma

Chroma також open-source і має Apache 2.0 license.

Вона дуже проста для локальних AI/RAG-прототипів і може працювати як local persistent database.

Chroma також має managed Cloud.

Я б використовувала Chroma для швидкого proof-of-concept, а Pinecone або Qdrant — коли навантаження та вимоги до production-середовища стають більшими.

---

# Чому обрана SPECTER2?

Для цього завдання використовується:

```text
allenai/specter2_base
```

SPECTER2 спеціально створений для representations наукових документів.

На відміну від універсальної моделі:

```text
all-MiniLM-L6-v2
```

яка орієнтована на загальний semantic similarity, SPECTER2 навчався на даних наукових публікацій та citation relationships.

За model card SPECTER2 навчався більш ніж на 6 мільйонах triplets наукових citation relationships.

У model card зазначені task formats:

```text
Classification
Regression
Proximity (Retrieval)
Adhoc Search
```

Тому domain-specific модель є більш логічним вибором для пошуку наукових статей.

## Важливе уточнення

Поточна Hugging Face model card описує `specter2_base` саме як базову модель, до якої для найкращої якості рекомендується підключати task-specific adapters.

Для general retrieval card рекомендує proximity/retrieval variant:

```text
allenai/specter2
```

Проте в цьому домашньому завданні за умовою використовується саме:

```text
allenai/specter2_base
```

через `sentence-transformers`, тому реалізація дотримується умов завдання.

Під час запуску бібліотека також вивела:

```text
No sentence-transformers model found with name allenai/specter2_base.
Creating a new one with mean pooling.
```

Це є одним з обмежень проведеного експерименту та може впливати на якість semantic search.

---

# Яку метрику рекомендує model card?

У поточній Hugging Face model card `allenai/specter2_base` не вказана явно одна обов'язкова метрика на кшталт `cosine`, `dot product` або `L2`.

Card більше фокусується на правильному використанні adapters та типі downstream task.

У цій роботі embeddings додатково нормалізуються:

```python
normalize_embeddings=True
```

а Pinecone index створено з:

```text
metric = cosine
```

Для нормалізованих векторів cosine similarity є природним вибором, а cosine і dot product стають еквівалентними за ranking.

---

# 1.3 Створення embeddings

Скрипт:

```text
scripts/02_embed.py
```

формує текст:

```text
title + " [SEP] " + abstract
```

і кодує 10 000 статей.

Використано:

```text
batch_size = 64
normalize_embeddings = True
```

Фактичний результат:

```text
Кількість ембеддингів: 10000
Розмірність: 768
Норма першого ембеддингу: 1.000000
Shape: (10000, 768)
```

Embeddings збережено у:

```text
embeddings/embeddings.npy
```

---

## Чому cosine similarity дорівнює dot product для нормалізованих vectors?

Cosine similarity:

```text
cos(x, y) = (x · y) / (||x|| ||y||)
```

Для нормалізованих vectors:

```text
||x|| = 1
||y|| = 1
```

Отже:

```text
cos(x, y) = x · y
```

Тобто cosine similarity та dot product мають однакові значення та однаковий ranking.

---

# Частина 2 — Завантаження у Pinecone

Скрипт:

```text
scripts/03_load_to_pinecone.py
```

створює index:

```text
arxiv-papers
```

Характеристики:

```text
dimension: 768
metric: cosine
```

Кожен vector містить metadata:

```text
arxiv_id
title
abstract
authors
year
category
```

Vectors завантажуються batches по:

```text
200
```

Фактичний результат після завантаження:

```text
total_vector_count: 10000
```

---

## Чому abstract обрізається?

До Pinecone metadata передається тільки:

```text
abstract[:500]
```

Повний abstract залишається у:

```text
data/arxiv_subset.parquet
```

Це зменшує metadata size у vector database.

Після пошуку повний текст документа можна отримати локально за його ID.

Такий підхід також логічно розділяє:

```text
vector index → retrieval
parquet → full document storage
```

---

# Частина 3 — Semantic Search

Скрипт:

```text
scripts/04_search.py
```

виконує:

1. semantic search у Pinecone;
2. semantic search з metadata filters;
3. локальне порівняння cosine, dot product та L2.

---

# Semantic search

Тестовий query:

```text
teaching machines to recognize objects in pictures
```

Приклад top result:

```text
Capturing knots in polymers
score ≈ 0.829
```

Результат не виглядає семантично ідеальним.

Причини:

1. наш subset складається з перших 10 000 записів arXiv;
2. статті переважно датовані 2007 роком;
3. потрібні сучасні ML-документи можуть бути відсутні;
4. `specter2_base` був завантажений через SentenceTransformer з fallback mean pooling.

Це важливе обмеження експерименту.

---

# Search з фільтрами

## Query A

```text
reinforcement learning
category = cs.LG
year >= 2021
```

Результат:

```text
Результатів не знайдено.
```

Це очікувано для нашої підмножини.

У subset практично всі 10 000 перших записів arXiv мають:

```text
year = 2007
```

Тому filter виключає всі documents.

---

## Query B

```text
machine learning methods
year < 2015
```

Top result:

```text
A neural network approach to ordinal regression
category: cs.LG
year: 2007
score ≈ 0.864
```

У цьому випадку filter відповідає характеристикам dataset, тому результати повертаються.

---

# Порівняння similarity metrics

Для query:

```text
teaching machines to recognize objects in pictures
```

отримано:

```text
Cosine top-5 indexes:
[378, 3350, 4115, 610, 3181]

Dot product top-5 indexes:
[378, 3350, 4115, 610, 3181]

L2 top-5 indexes:
[378, 3350, 4115, 610, 3181]
```

Фактичний результат:

```text
Cosine == Dot: True
Cosine == L2: True
```

---

## Чому cosine і dot product дали однаковий top-5?

Vectors були нормалізовані.

Для одиничних vectors:

```text
cos(x, y) = x · y
```

Тому ranking повністю збігається.

---

## Чому L2 також дав той самий ranking?

Для нормалізованих vectors:

```text
||x|| = ||y|| = 1
```

Тоді:

```text
||x - y||²
= ||x||² + ||y||² - 2(x · y)
= 2 - 2(x · y)
```

А оскільки:

```text
x · y = cosine(x, y)
```

отримуємо:

```text
L2² = 2 - 2 × cosine
```

Отже:

```text
maximum cosine
```

еквівалентний:

```text
minimum L2
```

для unit vectors.

Саме тому всі три методи дали однакове ранжування.

---

## Що було б без normalization?

Без normalization dot product залежав би не тільки від напряму vector, але й від його magnitude.

Документ з більшим vector norm міг би отримати високий dot product навіть при меншій семантичній близькості.

Cosine, навпаки, нормалізує vectors під час розрахунку.

Також перестала б діяти проста залежність:

```text
L2² = 2 - 2 × cosine
```

Тому rankings для cosine, dot product та L2 могли б відрізнятися.

---

# Частина 4 — Chunking

Скрипт:

```text
scripts/05_chunking.py
```

вибирає 30 статей з найдовшими abstracts.

Реалізовано дві стратегії.

---

## Fixed-size chunking

Параметри:

```text
chunk_size = 100 words
overlap = 20 words
```

Text ділиться кожні 100 words.

Наступний chunk повторює останні 20 words попереднього.

Фактичний результат:

```text
Fixed-size chunks: 120
```

---

## Semantic chunking

Текст спочатку розбивається на sentences.

Sentences додаються до chunk, доки total size не наближається до:

```text
100 words
```

Sentence не розрізається навмисно посередині.

Фактичний результат:

```text
Semantic chunks: 121
```

---

# Яка стратегія дає більш осмислені chunks?

Semantic chunking зазвичай дає більш цілісні fragments, тому що намагається не розривати sentence.

Наприклад, fragment:

```text
We believe that a comprehensive exploration on energy sources...
```

залишається логічно завершеною частиною тексту.

Fixed-size strategy орієнтується тільки на word count, тому semantic boundaries не враховуються.

---

# Чи можуть sentence розрізатися?

У fixed-size chunking — так.

100-word boundary може опинитися всередині sentence.

Тоді частина context переходить в інший chunk.

Це може погіршити embedding, тому що окремий chunk містить неповну думку.

Semantic chunking намагається уникати цього, групуючи цілі sentences.

---

# Вплив overlap

Overlap зменшує ризик втрати information на межі двох chunks.

У нашій реалізації:

```text
overlap = 20 words
```

Це означає, що частина context присутня у двох сусідніх chunks.

Перевага:

```text
краще покриття boundary context
```

Недолік:

```text
більша кількість chunks
більше embeddings
більше storage
більше duplicate content
```

---

# Результати chunk search

Було створено два Pinecone indexes:

```text
arxiv-chunks-fixed
arxiv-chunks-semantic
```

Результат:

```text
Fixed-size chunks: 120
Semantic chunks: 121
```

Пошук працював в обох indexes.

При цьому relevance була досить низькою для queries:

```text
machine learning methods
quantum systems and particles
```

Це пояснюється тим, що chunk indexes містять лише chunks з 30 найдовших abstracts, а не весь корпус.

Отже, пошукова система змушена ранжувати найкращі documents навіть тоді, коли серед 30 papers немає справді релевантного документа.

---

# Частина 5 — Hybrid Search

Скрипт:

```text
scripts/06_hybrid_search.py
```

реалізує три search methods:

```text
BM25
Vector Search
Hybrid Search
```

Hybrid search використовує:

```text
Reciprocal Rank Fusion
```

---

# BM25

BM25 працює на lexical matching.

Він особливо ефективний для:

- exact terminology;
- acronyms;
- names;
- identifiers;
- exact phrases.

---

# Vector Search

Vector search використовує embeddings.

Його сильна сторона:

- paraphrases;
- synonyms;
- natural-language questions;
- semantic similarity без exact token match.

---

# Reciprocal Rank Fusion

Для кожного document обчислюється:

```text
RRF score = Σ 1 / (k + rank)
```

Rankings BM25 та vector search об'єднуються в один список.

У роботі основне значення:

```text
k = 60
```

Також для порівняння:

```text
k = 1
```

---

# Порівняння результатів

## Query 1

```text
BERT fine-tuning
```

| Method | Top-1 |
|---|---|
| BM25 | A New Measure of Fine Tuning |
| Vector | Misere quotients for impartial games |
| Hybrid | A New Measure of Fine Tuning |

BM25 знайшов documents з exact phrase `fine tuning`, але це фізичні papers, а не BERT.

Причина проста: subset складається зі статей 2007 року, тоді як BERT з'явився значно пізніше.

Тому потрібного document фізично немає в corpus.

---

## Query 2

```text
Yann LeCun convolutional networks
```

| Method | Top-1 |
|---|---|
| BM25 | On Punctured Pragmatic Space-Time Codes in Block Fading Channel |
| Vector | Multilayer Perceptron with Functional Inputs |
| Hybrid k=60 | Optimization in Gradient Networks |

В corpus немає потрібного сучасного набору papers Yann LeCun.

Vector search принаймні знаходить paper про neural networks, тоді як BM25 значною мірою реагує на окремий token `networks`.

Hybrid при `k=60` підняв:

```text
Optimization in Gradient Networks
```

тому що цей document отримав contribution з обох ranking lists.

---

## Query 3

```text
making computers understand human emotions from text
```

| Method | Top-1 |
|---|---|
| BM25 | On the Development of Text Input Method - Lessons Learned |
| Vector | Opinion Dynamics and Sociophysics |
| Hybrid | On the Development of Text Input Method - Lessons Learned |

Цей приклад добре демонструє fusion.

Document:

```text
On the Development of Text Input Method - Lessons Learned
```

був:

```text
#1 BM25
#2 Vector Search
```

Тому Hybrid Search дав йому найбільший combined RRF score.

---

# Top-5 comparison

Фактичні IDs:

## BERT fine-tuning

```text
BM25:
[6243, 8389, 6984, 8717, 3658]

Vector:
[6406, 2535, 6795, 8935, 4441]

Hybrid:
[6243, 6406, 8389, 2535, 6984]
```

## Yann LeCun convolutional networks

```text
BM25:
[281, 1410, 1848, 1143, 391]

Vector:
[4213, 4821, 8935, 610, 7372]

Hybrid:
[1143, 281, 4213, 1410, 4821]
```

## Human emotions from text

```text
BM25:
[3664, 3661, 7897, 7321, 3710]

Vector:
[4893, 3664, 5681, 1157, 2541]

Hybrid:
[3664, 4893, 3661, 7897, 5681]
```

---

# Чи з'явилися в Hybrid top-5 documents, яких не було в top-5 окремих методів?

Фактичний результат для всіх трьох queries:

```text
Documents only in Hybrid top-5: []
```

Тобто в цьому експерименті RRF не додав у final top-5 document, який був би відсутній одночасно в top-5 BM25 та Vector Search.

Натомість він виконав:

```text
re-ranking
```

результатів двох методів.

Це також нормальний результат RRF: fusion не зобов'язаний створювати новий склад top-5 — його основне завдання полягає у зміні ranking на основі позицій documents у декількох списках.

---

# Як параметр k впливає на RRF?

Формула:

```text
1 / (k + rank)
```

При:

```text
k = 1
```

різниця між rank 1 та rank 5 дуже велика.

Тому перші позиції кожного окремого search method мають сильний вплив.

При:

```text
k = 60
```

різниця між сусідніми ranks значно менша.

Це дає більше значення documents, які стабільно присутні в обох rankings.

Наприклад, для:

```text
Yann LeCun convolutional networks
```

при `k=60` на перше місце вийшов:

```text
Optimization in Gradient Networks
```

тому що document отримав contribution з обох ranked lists.

При `k=1` top positions окремих methods мали набагато більшу вагу.

---

# Частина 6 — Аналіз та висновки

## 1. Semantic Search vs BM25

BM25 працює найкраще тоді, коли query містить exact terms.

Типові приклади:

```text
"BERT"
"Yann LeCun"
"fine-tuning"
```

За наявності таких terms у corpus lexical search може дуже точно знайти потрібний document.

Vector search краще підходить для queries у формі natural-language paraphrase.

Наприклад:

```text
making computers understand human emotions from text
```

не обов'язково має ті самі words, що й relevant scientific paper.

Embedding дозволяє знайти conceptual similarity.

У нашому конкретному експерименті жоден метод не міг дати ідеальний результат для всіх queries через обмеження corpus.

Перші 10 000 записів dataset переважно походять з 2007 року.

Тому запити про BERT або сучасні deep-learning papers фактично не мають правильного target document.

Загальне правило:

```text
exact entities / names / technical terms → BM25

paraphrases / semantic questions → Vector Search

mixed real-world search → Hybrid Search
```

---

# 2. Вплив розміру chunk

## Дуже маленький chunk

Наприклад:

```text
10–15 words
```

Перевага — локальна точність.

Але часто втрачається context.

Fragment може містити частину sentence без достатньої інформації для якісного embedding.

Також різко збільшується кількість vectors.

---

## Дуже великий chunk

Наприклад:

```text
500+ words
```

В одному embedding змішується багато різних concepts.

Semantic signal стає менш конкретним.

Також embedding model має context limit, тому text може бути truncated.

---

## Оптимальний розмір

Універсального оптимального chunk size не існує.

Він залежить від:

- типу document;
- embedding model;
- типу query;
- desired retrieval granularity;
- context window.

Для abstracts у цьому експерименті приблизно 100 words є розумним компромісом.

---

# 3. Що буде при Euclidean metric для normalized embeddings?

Для одиничних vectors:

```text
||x|| = ||y|| = 1
```

Euclidean distance:

```text
||x-y||²
=
||x||² + ||y||² - 2(x·y)
```

Тобто:

```text
||x-y||² = 2 - 2(x·y)
```

А для unit vectors:

```text
x·y = cosine(x,y)
```

Отже:

```text
L2² = 2 - 2 × cosine
```

З цього випливає:

```text
max cosine
⇔ max dot product
⇔ min L2
```

Тому для нормалізованих vectors використання Euclidean metric не повинно змінити ranking nearest neighbors.

Можуть відрізнятися самі numerical scores, але порядок documents залишається таким самим.

Це було підтверджено експериментально:

```text
Cosine == Dot: True
Cosine == L2: True
```

---

# 4. Pinecone Starter та масштабування

У цьому домашньому завданні фактично було успішно створено три indexes:

```text
arxiv-papers
arxiv-chunks-fixed
arxiv-chunks-semantic
```

Тобто під час виконання роботи обмеження на один index не виникло.

Актуальний Starter plan Pinecone дозволяє декілька indexes; умови тарифів можуть змінюватися з часом.

У нашому workload з 10 000 main vectors та приблизно 240 chunk vectors Starter був достатнім.

---

## Що робити з 10 мільйонами статей?

10 000 та 10 000 000 documents — це принципово різні масштаби.

Для 10 million papers я б:

1. не використовувала Starter tier;
2. перейшла на production Pinecone plan або distributed Qdrant deployment;
3. обробляла embeddings великими batches;
4. виконувала asynchronous ingestion;
5. зберігала raw documents окремо від vector DB;
6. залишала у metadata лише поля, потрібні для retrieval та filtering;
7. використовувала monitoring latency, storage та ingestion throughput;
8. розділила corpus через namespaces, logical partitions або collections;
9. використовувала incremental indexing замість повної перебудови;
10. тестувала ANN parameters та recall на representative evaluation dataset.

Vector database у такій архітектурі повинна використовуватися насамперед як retrieval layer, а не як основне storage для повних scientific documents.

---

# Обмеження експерименту

Результати цього домашнього завдання мають кілька важливих обмежень.

## Dataset selection

Було взято перші:

```text
10000
```

valid records arXiv dataset.

Вони переважно датовані:

```text
2007
```

Тому modern AI queries мають слабке coverage.

---

## SPECTER2 integration

За умовою завдання використано:

```text
SentenceTransformer("allenai/specter2_base")
```

Поточна версія `sentence-transformers` вивела warning:

```text
No sentence-transformers model found with name allenai/specter2_base.
Creating a new one with mean pooling.
```

При цьому embeddings коректно створилися:

```text
10000 embeddings
dimension = 768
norm = 1.0
```

але це слід враховувати під час оцінки retrieval quality.

У production implementation я б протестувала офіційний SPECTER2 retrieval/proximity adapter та порівняла його quality з поточною реалізацією.

---

## Chunking experiment

Chunk search виконувався тільки по:

```text
30 longest abstracts
```

Тому його relevance не можна порівнювати напряму з пошуком по всьому corpus.

Мета цієї частини — насамперед продемонструвати різницю між chunking strategies.

---

# Підсумок

У межах домашнього завдання було:

- підготовлено 10 000 scientific papers;
- створено 768-dimensional embeddings;
- нормалізовано embeddings;
- завантажено 10 000 vectors у Pinecone;
- реалізовано semantic search;
- реалізовано metadata filtering;
- порівняно cosine similarity, dot product та L2;
- реалізовано fixed-size chunking;
- реалізовано semantic chunking;
- створено окремі chunk indexes;
- реалізовано BM25;
- реалізовано vector search;
- реалізовано Reciprocal Rank Fusion;
- порівняно `k=60` та `k=1`;
- проаналізовано обмеження dataset, embeddings та vector database.

Головний практичний висновок:

```text
BM25 знаходить lexical matches.
Vector Search знаходить semantic matches.
RRF дозволяє об'єднати обидва signals.
```

Для production search найкращою практикою є не вибір одного універсального методу, а комбінація lexical та semantic retrieval з подальшим re-ranking.