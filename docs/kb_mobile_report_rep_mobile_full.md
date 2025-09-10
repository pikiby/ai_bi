---
id: table_mobile_report_rep_mobile_full
title: Оплаты мобильного приложения в день
db: db1
short_description: "Ежедневная статистика оплат в мобильном приложении по платформам и ценовым категориям; включает возвраты (App Store и ЮKassa). Для месячных отчётов суммируйте значения за дни месяца."
synonyms:
- оплаты мобильного приложения
- платежи App Store
- платежи ЮKassa
- мобильный iOS
- мобильный Android
type: table
---
# «Оплаты мобильного приложения в день» — db1.mobile_report_rep_mobile_full

## Названия таблицы

**Короткое имя (человекочитаемое):** Оплаты мобильного приложения в день  
**Тех. имя:** `db1.mobile_report_rep_mobile_full`

## Назначение
Ежедневная статистика оплат в мобильном приложении по платформам и ценовым категориям, включая возвраты (App Store и ЮKassa). Для месячных отчётов суммируйте значения за дни месяца.

**Терминология:** под «сумма оплат» понимается выражение `IOS_PL + Android_PL`.


## Хранилище и движок
- БД: `db1`
- Таблица: `mobile_report_rep_mobile_full`
- Движок: `MergeTree`
- Сортировка: `ORDER BY partner_uuid`

## DDL
```sql
CREATE TABLE db1.mobile_report_rep_mobile_full
(
    `report_date` Date,
    `partner_uuid` String,
    `city` String,
    `IOS_PL` Int64,
    `appstore_count_85` UInt64,
    `appstore_count_85_refunded` UInt64,
    `appstore_count_69` UInt64,
    `appstore_count_69_refunded` UInt64,
    `appstore_count_499` UInt64,
    `appstore_count_499_refunded` UInt64,
    `appstore_count_2390` UInt64,
    `appstore_count_2390_refunded` UInt64,
    `appstore_count_1` UInt64,
    `appstore_count_1_refunded` UInt64,
    `refunded_amount_appstore` Int64,
    `Android_PL` Int64,
    `yookassa_count_85` UInt64,
    `yookassa_count_85_refunded` UInt64,
    `yookassa_count_69` UInt64,
    `yookassa_count_69_refunded` UInt64,
    `yookassa_count_35` UInt64,
    `yookassa_count_35_refunded` UInt64,
    `yookassa_count_1` UInt64,
    `yookassa_count_1_refunded` UInt64,
    `yookassa_count_499` UInt64,
    `yookassa_count_499_refunded` UInt64,
    `yookassa_count_249` UInt64,
    `yookassa_count_249_refunded` UInt64,
    `yookassa_count_2390` UInt64,
    `yookassa_count_2390_refunded` UInt64,
    `refunded_amount_yookassa` Int64
)
ENGINE = MergeTree()
ORDER BY partner_uuid
```

## Поля и алиасы
| Поле                           | Тип      | Алиас (человекочитаемое)             |
| ------------------------------ | -------- | ------------------------------------ |
| `report_date`                  | `Date`   | `дата отчёта`                        |
| `partner_uuid`                 | `String` | `идентификатор партнёра`             |
| `city`                         | `String` | `название города`                    |
| `IOS_PL`                       | `Int64`  | `iOS: сумма платежей за день`        |
| `appstore_count_85`            | `UInt64` | `App Store: покупок 85 за день`      |
| `appstore_count_85_refunded`   | `UInt64` | `App Store: возвратов 85 за день`    |
| `appstore_count_69`            | `UInt64` | `App Store: покупок 69 за день`      |
| `appstore_count_69_refunded`   | `UInt64` | `App Store: возвратов 69 за день`    |
| `appstore_count_499`           | `UInt64` | `App Store: покупок 499 за день`     |
| `appstore_count_499_refunded`  | `UInt64` | `App Store: возвратов 499 за день`   |
| `appstore_count_2390`          | `UInt64` | `App Store: покупок 2390 за день`    |
| `appstore_count_2390_refunded` | `UInt64` | `App Store: возвратов 2390 за день`  |
| `appstore_count_1`             | `UInt64` | `App Store: покупок 1 за день`       |
| `appstore_count_1_refunded`    | `UInt64` | `App Store: возвратов 1 за день`     |
| `refunded_amount_appstore`     | `Int64`  | `App Store: сумма возвратов за день` |
| `Android_PL`                   | `Int64`  | `Android: сумма платежей за день`    |
| `yookassa_count_85`            | `UInt64` | `ЮKassa: оплат 85 за день`           |
| `yookassa_count_85_refunded`   | `UInt64` | `ЮKassa: возвратов 85 за день`       |
| `yookassa_count_69`            | `UInt64` | `ЮKassa: оплат 69 за день`           |
| `yookassa_count_69_refunded`   | `UInt64` | `ЮKassa: возвратов 69 за день`       |
| `yookassa_count_35`            | `UInt64` | `ЮKassa: оплат 35 за день`           |
| `yookassa_count_35_refunded`   | `UInt64` | `ЮKassa: возвратов 35 за день`       |
| `yookassa_count_1`             | `UInt64` | `ЮKassa: оплат 1 за день`            |
| `yookassa_count_1_refunded`    | `UInt64` | `ЮKassa: возвратов 1 за день`        |
| `yookassa_count_499`           | `UInt64` | `ЮKassa: оплат 499 за день`          |
| `yookassa_count_499_refunded`  | `UInt64` | `ЮKassa: возвратов 499 за день`      |
| `yookassa_count_249`           | `UInt64` | `ЮKassa: оплат 249 за день`          |
| `yookassa_count_249_refunded`  | `UInt64` | `ЮKassa: возвратов 249 за день`      |
| `yookassa_count_2390`          | `UInt64` | `ЮKassa: оплат 2390 за день`         |
| `yookassa_count_2390_refunded` | `UInt64` | `ЮKassa: возвратов 2390 за день`     |
| `refunded_amount_yookassa`     | `Int64`  | `ЮKassa: сумма возвратов за день`    |

> Примечание к ценам 85/69/499/2390/249/35/1: единицы соответствуют исходным транзакциям; при необходимости уточните валюту/номинал.

## Частые срезы/фильтры
- Сумма оплат = IOS_PL + ANDROID_PL
- По дате: `report_date`
- По партнёру: `partner_uuid`
- По городу: `city`
- По платформе: `IOS_PL`, `Android_PL`
- По провайдеру/номиналу: поля `appstore_count_*`, `yookassa_count_*`

## Примеры запросов (с алиасами)


4) Сумма оплат за месяц (агрегирование по дням):
```sql
WITH month_bounds AS (
  SELECT toDate('2025-08-01') AS m_start, addMonths(toDate('2025-08-01'), 1) AS m_end
)
SELECT
  t.partner_uuid AS `идентификатор партнёра`,
  sum(t.IOS_PL + t.Android_PL) AS `сумма оплат за месяц`
FROM db1.mobile_report_rep_mobile_full AS t
CROSS JOIN month_bounds
WHERE t.report_date >= m_start AND t.report_date < m_end
GROUP BY t.partner_uuid;
```

1) Сумма оплат за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.mobile_report_rep_mobile_full
)
SELECT
  t.partner_uuid AS `идентификатор партнёра`,
  (t.IOS_PL + t.Android_PL) AS `сумма оплат за день`
FROM db1.mobile_report_rep_mobile_full AS t
INNER JOIN max_dt USING(report_date);
```


2) Месячный отчёт (сумма за август 2025):
```sql
WITH month_bounds AS (
  SELECT toDate('2025-08-01') AS m_start, addMonths(toDate('2025-08-01'), 1) AS m_end
)
SELECT
  t.partner_uuid AS `идентификатор партнёра`,
  sum(t.IOS_PL)     AS `iOS: сумма платежей за месяц`,
  sum(t.Android_PL) AS `Android: сумма платежей за месяц`
FROM db1.mobile_report_rep_mobile_full AS t
CROSS JOIN month_bounds
WHERE t.report_date >= m_start AND t.report_date < m_end
GROUP BY t.partner_uuid;
```

3) App Store: топ-10 городов по покупкам 499 за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.mobile_report_rep_mobile_full
)
SELECT
  t.city AS `название города`,
  t.appstore_count_499 AS `App Store: покупок 499 за день`
FROM db1.mobile_report_rep_mobile_full AS t
INNER JOIN max_dt USING(report_date)
ORDER BY 2 DESC
LIMIT 10;
```

## Ключи и соединения
- Сортировка `ORDER BY partner_uuid`. Используйте `partner_uuid` для группировок/фильтров; для городских срезов — `city`.

## Ограничения и примечания
- Данные — **за день** (не накопительные). Для месячных значений агрегируйте по датам месяца (например, `sum(...)` с фильтром по месяцу).
- NULL-политика/дефолты и источники данных не описаны.
- Партиционирование не указано в DDL.
- Это b2c платежи. Не b2b

<!-- ai-ignore:start -->
Tags: #KnowledgeBase
Links: [[mobile_report_rep_mobile_full]]
<!-- ai-ignore:end -->