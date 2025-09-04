# «Оплаты мобильного приложения в месяц (накопительно)» — db1.mobile_report_cum_rep_mobile_full

## Названия таблицы

**Короткое имя (человекочитаемое):** Оплаты мобильного приложения в месяц (накопительно)  
**Тех. имя:** `db1.mobile_report_cum_rep_mobile_full`

## Назначение
Накопительная (cum) статистика **внутри календарного месяца** по оплатам в мобильном приложении по платформам и ценовым категориям, включая возвраты (App Store и ЮKassa).

**Как читать:** для отчёта за месяц используйте **последнюю доступную дату этого месяца** (month-end), без разностей между днями.

## Хранилище и движок
- БД: `db1`
- Таблица: `mobile_report_cum_rep_mobile_full`
- Движок: `MergeTree`
- Сортировка: `ORDER BY partner_uuid`

## DDL
```sql
CREATE TABLE db1.mobile_report_cum_rep_mobile_full
(
    `report_date` Date,
    `partner_uuid` String,
    `city` String,
    `IOS_PL_cum` Int64,
    `appstore_count_85_cum` UInt64,
    `appstore_count_85_refunded_cum` UInt64,
    `appstore_count_69_cum` UInt64,
    `appstore_count_69_refunded_cum` UInt64,
    `appstore_count_499_cum` UInt64,
    `appstore_count_499_refunded_cum` UInt64,
    `appstore_count_2390_cum` UInt64,
    `appstore_count_2390_refunded_cum` UInt64,
    `appstore_count_1_cum` UInt64,
    `appstore_count_1_refunded_cum` UInt64,
    `refunded_amount_appstore_1_cum` Int64,
    `Android_PL_cum` Int64,
    `yookassa_count_85_cum` UInt64,
    `yookassa_count_85_refunded_cum` UInt64,
    `yookassa_count_69_cum` UInt64,
    `yookassa_count_69_refunded_cum` UInt64,
    `yookassa_count_35_cum` UInt64,
    `yookassa_count_35_refunded_cum` UInt64,
    `yookassa_count_1_cum` UInt64,
    `yookassa_count_1_refunded_cum` UInt64,
    `yookassa_count_499_cum` UInt64,
    `yookassa_count_499_refunded_cum` UInt64,
    `yookassa_count_249_cum` UInt64,
    `yookassa_count_249_refunded_cum` UInt64,
    `yookassa_count_2390_cum` UInt64,
    `yookassa_count_2390_refunded_cum` UInt64,
    `refunded_amount_yookassa_cum` Int64
)
ENGINE = MergeTree()
ORDER BY partner_uuid
```

## Поля и алиасы
| Поле                                 | Тип     | Алиас (человекочитаемое)                                             |
|---                                   |---      |---                                                                   |
| `report_date`                        | `Date`  | `дата отчёта`                                                        |
| `partner_uuid`                       | `String`| `идентификатор партнёра`                                             |
| `city`                               | `String`| `название города`                                                    |
| `IOS_PL_cum`                         | `Int64` | `iOS: сумма платежей, накопительно`                                  |
| `appstore_count_85_cum`              | `UInt64`| `App Store: покупок по цене 85, накопительно`                        |
| `appstore_count_85_refunded_cum`     | `UInt64`| `App Store: возвратов по цене 85, накопительно`                      |
| `appstore_count_69_cum`              | `UInt64`| `App Store: покупок по цене 69, накопительно`                        |
| `appstore_count_69_refunded_cum`     | `UInt64`| `App Store: возвратов по цене 69, накопительно`                      |
| `appstore_count_499_cum`             | `UInt64`| `App Store: покупок по цене 499, накопительно`                       |
| `appstore_count_499_refunded_cum`    | `UInt64`| `App Store: возвратов по цене 499, накопительно`                     |
| `appstore_count_2390_cum`            | `UInt64`| `App Store: покупок по цене 2390, накопительно`                      |
| `appstore_count_2390_refunded_cum`   | `UInt64`| `App Store: возвратов по цене 2390, накопительно`                    |
| `appstore_count_1_cum`               | `UInt64`| `App Store: покупок по цене 1, накопительно`                         |
| `appstore_count_1_refunded_cum`      | `UInt64`| `App Store: возвратов по цене 1, накопительно`                       |
| `refunded_amount_appstore_1_cum`     | `Int64` | `App Store: сумма возвратов по цене 1, накопительно`                 |
| `Android_PL_cum`                     | `Int64` | `Android: сумма платежей, накопительно`                              |
| `yookassa_count_85_cum`              | `UInt64`| `ЮKassa: оплат по цене 85, накопительно`                             |
| `yookassa_count_85_refunded_cum`     | `UInt64`| `ЮKassa: возвратов по цене 85, накопительно`                         |
| `yookassa_count_69_cum`              | `UInt64`| `ЮKassa: оплат по цене 69, накопительно`                             |
| `yookassa_count_69_refunded_cum`     | `UInt64`| `ЮKassa: возвратов по цене 69, накопительно`                         |
| `yookassa_count_35_cum`              | `UInt64`| `ЮKassa: оплат по цене 35, накопительно`                             |
| `yookassa_count_35_refunded_cum`     | `UInt64`| `ЮKassa: возвратов по цене 35, накопительно`                         |
| `yookassa_count_1_cum`               | `UInt64`| `ЮKassa: оплат по цене 1, накопительно`                              |
| `yookassa_count_1_refunded_cum`      | `UInt64`| `ЮKassa: возвратов по цене 1, накопительно`                          |
| `yookassa_count_499_cum`             | `UInt64`| `ЮKassa: оплат по цене 499, накопительно`                            |
| `yookassa_count_499_refunded_cum`    | `UInt64`| `ЮKassa: возвратов по цене 499, накопительно`                        |
| `yookassa_count_249_cum`             | `UInt64`| `ЮKassa: оплат по цене 249, накопительно`                            |
| `yookassa_count_249_refunded_cum`    | `UInt64`| `ЮKassa: возвратов по цене 249, накопительно`                        |
| `yookassa_count_2390_cum`            | `UInt64`| `ЮKassa: оплат по цене 2390, накопительно`                           |
| `yookassa_count_2390_refunded_cum`   | `UInt64`| `ЮKassa: возвратов по цене 2390, накопительно`                       |
| `refunded_amount_yookassa_cum`       | `Int64` | `ЮKassa: сумма возвратов, накопительно`                              |

> Примечание к ценам 85/69/499/2390/249/35/1: единицы (валюта/номинал) соответствуют исходным транзакциям; при необходимости уточните.

## Частые срезы/фильтры
- По дате: `report_date`
- По партнёру: `partner_uuid`
- По городу: `city`
- По платформе: агрегаты `IOS_PL_cum`, `Android_PL_cum`
- По платёжному провайдеру/номиналу: поля с префиксами `appstore_count_*` и `yookassa_count_*`

## Примеры запросов (с алиасами)


0) Отчёт за конкретный месяц (например, август 2025):
```sql
WITH month_bounds AS (
  SELECT toDate('2025-08-01') AS m_start, addMonths(toDate('2025-08-01'), 1) AS m_end
),
last_day AS (
  SELECT max(t.report_date) AS report_date
  FROM db1.mobile_report_cum_rep_mobile_full AS t
  CROSS JOIN month_bounds
  WHERE t.report_date >= m_start AND t.report_date < m_end
)
SELECT
  t.partner_uuid AS `идентификатор партнёра`,
  t.IOS_PL_cum     AS `iOS: сумма платежей (накопительно за месяц)`,
  t.Android_PL_cum AS `Android: сумма платежей (накопительно за месяц)`
FROM db1.mobile_report_cum_rep_mobile_full AS t
INNER JOIN last_day USING(report_date);
```

1) Итог по платформам за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.mobile_report_cum_rep_mobile_full
)
SELECT
  t.partner_uuid AS `идентификатор партнёра`,
  t.IOS_PL_cum     AS `iOS: сумма платежей (накопительно)`,
  t.Android_PL_cum AS `Android: сумма платежей (накопительно)`
FROM db1.mobile_report_cum_rep_mobile_full AS t
INNER JOIN max_dt USING(report_date)
```

2) Разбивка App Store по городам (последняя дата), топ-10 по покупкам 499:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.mobile_report_cum_rep_mobile_full
)
SELECT
  t.city AS `название города`,
  t.appstore_count_499_cum AS `App Store: покупок 499 (накоп.)`,
  t.appstore_count_499_refunded_cum AS `App Store: возвратов 499 (накоп.)`
FROM db1.mobile_report_cum_rep_mobile_full AS t
INNER JOIN max_dt USING(report_date)
ORDER BY 2 DESC
LIMIT 10
```

3) Доля возвратов по провайдерам (последняя дата):
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.mobile_report_cum_rep_mobile_full
),
agg AS (
  SELECT
    sum(appstore_count_85_refunded_cum + appstore_count_69_refunded_cum + appstore_count_499_refunded_cum
      + appstore_count_2390_refunded_cum + appstore_count_1_refunded_cum) AS appstore_refunds,
    sum(yookassa_count_85_refunded_cum + yookassa_count_69_refunded_cum + yookassa_count_35_refunded_cum
      + yookassa_count_1_refunded_cum + yookassa_count_499_refunded_cum + yookassa_count_249_refunded_cum
      + yookassa_count_2390_refunded_cum) AS yookassa_refunds
  FROM db1.mobile_report_cum_rep_mobile_full
  INNER JOIN max_dt USING(report_date)
)
SELECT
  appstore_refunds AS `App Store: возвраты (накоп.)`,
  yookassa_refunds AS `ЮKassa: возвраты (накоп.)`
FROM agg
```

## Ключи и соединения
- Сортировка `ORDER BY partner_uuid`. Используйте `partner_uuid` для группировок/фильтров; для городских срезов — `city`.

## Ограничения и примечания
- Поля имеют накопительный характер ( `*_cum` ) **в пределах месяца**; для значения «за месяц» берите строку с максимальной датой в этом месяце (`max(report_date)` при фильтре по месяцу). Разности по датам считать **не нужно**.
- NULL-политика/дефолты и источники данных не описаны./дефолты и источники данных не описаны.
- Партиционирование не указано в DDL.