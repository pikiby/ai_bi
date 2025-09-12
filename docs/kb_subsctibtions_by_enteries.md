---
id: table_t_subsctibtions_by_enteries
title: Подписки и квартиры по подъездам
db: db1
short_description: "Дневной срез подписок по подъездам: фактическое и партнёрское число квартир, доля подписок, разрезы по адресу, городу и партнёру."
synonyms:
  - подписки по подъездам
  - квартиры и подписки
  - адресные подписки
  - проникновение подписки
type: table
source:
  - "[[t_subsctibtions_by_enteries]]"
tags:
  - "#KnowledgeBase"
---

# «Подписки и квартиры по подъездам» — t_subsctibtions_by_enteries

## Названия таблицы
**Короткое имя (человекочитаемое):** Подписки и Квартиры по подъездам  
**Тех. имя:** `t_subsctibtions_by_enteries`

## Назначение
Таблица даёт ежедневный срез по подъездам (unit = `address_uuid`): число квартир, число подписок и показатели проникновения (по фактическому и партнёрскому диапазону квартир). Используется для аналитики проникновения, контроля качества справочников и мониторинга партнёров.

**Определения и формулы:**  
- `subscribtion_rate = subscribed_citizen_id_count / flats_count * 100`  
- `subscribtion_rate_range = subscribed_citizen_id_count / flats_count_range * 100`  
Рекомендуется безопасное деление: `x / nullIf(y, 0)`.

## Хранилище и движок
- БД: `db1`  
- Таблица: `t_subsctibtions_by_enteries`  
- Движок: `MergeTree`  
- Сортировка: `ORDER BY report_date`

## DDL
```sql
CREATE TABLE t_subsctibtions_by_enteries
(
    `report_date` Date,
    `full_address` String,
    `city` String,
    `installation_point_id` Int64,
    `flats_count` Int16,
    `flats_count_range` Int16,
    `address_uuid` String,
    `partner_uuid` String,
    `subscribed_citizen_id_count` UInt64,
    `tariff_full` String,
    `company_name` String,
    `tin` String,
    `partner_lk` String,
    `subscribtion_rate` Float64,
    `subscribtion_rate_range` Float64
)
ENGINE = MergeTree()
ORDER BY report_date;
```

## Поля и алиасы
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `report_date`                 | Date    | Дата                                      |
| `full_address`                | String  | Полный адрес                              |
| `city`                        | String  | Город                                     |
| `installation_point_id`       | Int64   | ID точки установки/узла                   |
| `flats_count`                 | Int16   | Количество квартир по БД                  |
| `flats_count_range`           | Int16   | Количество квартир (диапазон от партнёра) |
| `address_uuid`                | String  | Идентификатор подъезда (главный юнит)     |
| `partner_uuid`                | String  | Идентификатор партнёра                    |
| `subscribed_citizen_id_count` | UInt64  | Подписок в подъезде                       |
| `tariff_full`                 | String  | Тариф (полное наименование)               |
| `company_name`                | String  | Название фирмы                            |
| `tin`                         | String  | ИНН                                       |
| `partner_lk`                  | String  | Личный кабинет                            |
| `subscribtion_rate`           | Float64 | Доля подписанных от `flats_count`         |
| `subscribtion_rate_range`     | Float64 | Доля подписанных от `flats_count_range`   |

## Примечания к полям:  
- `company_name` - это название компании, партнеры, фирмы, названия личных кабинетов.
- `partner_lk` - это личный кабинет партнеров, личный кабинет компаний, ЛК
- `subscribtion_rate` - процент процент проникновения в подъезд, Доля подписанных от flats_count в подъезде
- `address_uuid` — **главный юнит**, адрес конкретного подъезда.  
- `flats_count` — количество квартир по базе данных.  
- `flats_count_range` задаётся партнёром и может отличаться от факта.  
- Используйте безопасное деление: `subscribed_citizen_id_count / nullIf(flats_count, 0)` и аналогично для `flats_count_range`.

## Частые срезы/фильтры
- По дате (`report_date`), чаще всего — **последняя доступная дата**.  
- По подъезду (`address_uuid`) — основной ключ аналитики.  
- По партнёру (`partner_uuid`, `company_name`, `tin`, `partner_lk`).  
- По городу (`city`).  
- По тарифу (`tariff_full`).  
- Контроль качества данных: исключайте строки с `flats_count <= 0` или `flats_count_range <= 0` при расчёте долей.

### Определение «последней даты»
Последняя дата — это максимальная `report_date`, для которой по всей таблице выполняется условие:  
`sum(subscribed_citizen_id_count) > 0`.  
Дни с нулевыми подписками (возможные «пустые» проводы или будущие даты) игнорируются.

## Примеры запросов (ClickHouse)

1) **Топ-20 подъездов по доле подписки (последняя дата, факт квартир)**
```sql
WITH last_dt AS (
  SELECT report_date
  FROM t_subsctibtions_by_enteries
  GROUP BY report_date
  HAVING sum(subscribed_citizen_id_count) > 0
  ORDER BY report_date DESC
  LIMIT 1
)
SELECT
  t.address_uuid                           AS `подъезд`,
  anyHeavy(t.full_address)                 AS `адрес`,
  anyHeavy(t.city)                         AS `город`,
  sum(t.subscribed_citizen_id_count)       AS `подписок`,
  sum(t.flats_count)                       AS `квартир (факт)`,
  `подписок` / nullIf(`квартир (факт)`, 0) AS `доля подписки`
FROM t_subsctibtions_by_enteries AS t
INNER JOIN last_dt USING (report_date)
GROUP BY t.address_uuid
HAVING `квартир (факт)` > 0
ORDER BY `доля подписки` DESC
LIMIT 20;
```

2) **Сводка по партнёрам за месяц (факт и диапазон)**
```sql
WITH bounds AS (
  SELECT toDate('2025-08-01') AS d1, addMonths(toDate('2025-08-01'), 1) AS d2
)
SELECT
  t.partner_uuid                                           AS `партнёр`,
  anyHeavy(t.company_name)                                 AS `юрлицо`,
  sum(t.subscribed_citizen_id_count)                       AS `подписок за месяц`,
  sum(t.flats_count)                                       AS `квартир (факт)`,
  sum(t.flats_count_range)                                 AS `квартир (диапазон)`,
  `подписок за месяц` / nullIf(`квартир (факт)`, 0)        AS `доля (факт)`,
  `подписок за месяц` / nullIf(`квартир (диапазон)`, 0)    AS `доля (диапазон)`
FROM t_subsctibtions_by_enteries AS t
CROSS JOIN bounds
WHERE t.report_date >= d1 AND t.report_date < d2
GROUP BY t.partner_uuid;
```

3) **Контроль расхождений: где диапазон сильно отличается от факта (последняя дата)**
```sql
WITH last_dt AS (
  SELECT report_date
  FROM t_subsctibtions_by_enteries
  GROUP BY report_date
  HAVING sum(subscribed_citizen_id_count) > 0
  ORDER BY report_date DESC
  LIMIT 1
)
SELECT
  t.address_uuid,
  anyHeavy(t.full_address) AS `адрес`,
  anyHeavy(t.city)         AS `город`,
  sum(t.flats_count)       AS `факт`,
  sum(t.flats_count_range) AS `диапазон`,
  abs(`диапазон` - `факт`) AS `дельта`,
  `дельта` / nullIf(`факт`, 0) AS `отн_дельта`
FROM t_subsctibtions_by_enteries AS t
INNER JOIN last_dt USING (report_date)
GROUP BY t.address_uuid
HAVING `факт` > 0 AND `диапазон` > 0 AND `отн_дельта` >= 0.2
ORDER BY `дельта` DESC
LIMIT 50;
```

4) **Топ-10 городов по проникновению (последняя дата, по факту)**
```sql
WITH last_dt AS (
  SELECT report_date
  FROM t_subsctibtions_by_enteries
  GROUP BY report_date
  HAVING sum(subscribed_citizen_id_count) > 0
  ORDER BY report_date DESC
  LIMIT 1
)
SELECT
  t.city                             AS `город`,
  sum(t.subscribed_citizen_id_count) AS `подписок`,
  sum(t.flats_count)                 AS `квартир`,
  `подписок` / nullIf(`квартир`, 0)  AS `доля`
FROM t_subsctibtions_by_enteries AS t
INNER JOIN last_dt USING (report_date)
GROUP BY t.city
HAVING `квартир` > 0
ORDER BY `доля` DESC
LIMIT 10;
```

5) **История по конкретному подъезду**
```sql
SELECT
  report_date,
  subscribed_citizen_id_count                 AS subscriptions,
  flats_count                                 AS flats_fact,
  flats_count_range                           AS flats_range,
  subscriptions / nullIf(flats_fact, 0)       AS rate_fact,
  subscriptions / nullIf(flats_range, 0)      AS rate_range
FROM t_subsctibtions_by_enteries
WHERE address_uuid = '<<ADDRESS_UUID>>'
ORDER BY report_date;
```

## Ключи и соединения
- **Основной аналитический ключ:** `address_uuid` (подъезд).  
- Возможные связи: к справочнику адресов/домов — по `address_uuid` или `installation_point_id`; к партнёрам/юрлицам — по `partner_uuid`/`tin`.

## Ограничения и примечания
- Данные — **за день** (значения не накопительные). Для месячных показателей аккумулируйте по датам.  
- `flats_count` и `flats_count_range` могут быть нулевыми — используйте `nullIf(...)` в формулах.  
- `flats_count_range` — оценка партнёра; контролируйте аномалии и расхождения.  
- Возможны задержки/пропуски атрибутов (`tariff_full`, `partner_lk`).