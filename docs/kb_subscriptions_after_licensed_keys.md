---
id: t_subscriptions_after_licensed_keys
db: db1
short_description: "Дневной срез метрик по подпискам и лицензиям: активированные ключи, подписки с ключом (возвраты/новые) и их отмены в разрезе партнёров и городов."
synonyms:
  - мобильные подписки
  - ключи
  - подписки после лицензии
  - возвраты с ключом
  - новые с ключом
type: table
tags:
  - "#KnowledgeBase"
---

# «Подписки после приобретения ключей» — t_subscriptions_after_licensed_keys

## Назначение
Таблица фиксирует ежедневный срез метрик, связанных с приобретением лицензионных ключей и подписками, в разрезе партнёров и городов. Включает количество активированных ключей, количество подписок, оформленных с участием ключа (возвраты и новые), а также их первые отмены.

## Хранилище и движок
- БД: `db1`
- Таблица: `t_subscriptions_after_licensed_keys`
- Движок: `MergeTree`
- Сортировка: `ORDER BY (report_date, partner_uuid, city_uuid)`

## DDL
```sql
CREATE TABLE IF NOT EXISTS db1.t_subscriptions_after_licensed_keys
(
    `report_date` Date,
    `partner_uuid` String,
    `city_uuid` String,
    `activated_keys` UInt32,
    `activated_sub_w_key` UInt32,
    `created_first_sub_w_key` UInt32,
    `d_activated_sub_w_key` UInt32,
    `d_created_first_sub_w_key` UInt32
)
ENGINE = MergeTree()
ORDER BY (report_date, partner_uuid, city_uuid)
```

### Материализованное представление
Обновляется ежедневно и заполняет таблицу результирующими метриками.
```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS db1.t_subscriptions_after_licensed_keys_mv
REFRESH EVERY 1 DAY OFFSET 5 HOUR 33 MINUTE TO db1.t_subscriptions_after_licensed_keys AS
-- см. реализацию в ноутбуке `t_subscriptions_after_licensed_keys.ipynb`
-- включает расчёт:
--  - activated_keys
--  - activated_sub_w_key
--  - created_first_sub_w_key
--  - d_activated_sub_w_key
--  - d_created_first_sub_w_key
SELECT ...
```

## Поля и алиасы
| Поле                         | Тип    | Алиас (человекочитаемое)                     |
|-----------------------------|--------|----------------------------------------------|
| `report_date`               | Date   | Дата                                         |
| `partner_uuid`              | String | UUID партнёра                                |
| `city_uuid`                 | String | UUID города                                  |
| `activated_keys`            | UInt32 | Активированные ключи                          |
| `activated_sub_w_key`       | UInt32 | Вернувшиеся подписчики с ключем               |
| `created_first_sub_w_key`   | UInt32 | Новые подписчики с ключем                     |
| `d_activated_sub_w_key`     | UInt32 | Отменившие подписку (вернувшиеся)             |
| `d_created_first_sub_w_key` | UInt32 | Отменившие подписку (новые)                   |

## Примечания
- Под «с ключом» понимаются события, произошедшие в окне ±48 часов относительно активации/создания подписки и времени приобретения ключа (в зависимости от сценария).
- Для расчёта используются: `citizens_dir_mobile_ch`, `subscriptions_st_mobile_ch`, `licensed_keys_dir_partner_ch`, `t_a_houses_by_partner`, а также связи дом → город из `entries_installation_points_dir_partner_ch`.
- Метрики отмен (`d_*`) учитывают первую деактивацию после появления соответствующего события у пользователя.

## Связи и обогащение
- Город (человекочитаемо): из `t_a_city_uuid`
  - Ключи связи: `city_uuid`
  - Поле: `city` → алиас `Город`
- Реквизиты компании: из `companies_dir_partner_ch`
  - Ключ связи: `partner_uuid`
  - Поля:
    - `company_name` → алиас `Компания`
    - `partner_lk` → алиас `ЛК партнёра`
    - `tin` → алиас `ИНН`

### Пример агрегированного отчёта по последней дате (с обогащением)
```sql
WITH s AS (
    SELECT
        `report_date`,
        `partner_uuid`,
        `city_uuid`,
        `activated_keys`,
        `activated_sub_w_key`,
        `created_first_sub_w_key`,
        `d_activated_sub_w_key`,
        `d_created_first_sub_w_key`
    FROM `t_subscriptions_after_licensed_keys`
    WHERE `report_date` = (
        SELECT max(`report_date`) FROM `t_subscriptions_after_licensed_keys`
    )
)
SELECT
    s.`report_date`                              AS `Дата`,
    s.`partner_uuid`                             AS `UUID партнёра`,
    s.`city_uuid`                                AS `UUID города`,
    e.`city`                                     AS `Город`,
    c.`company_name`                             AS `Компания`,
    c.`partner_lk`                               AS `ЛК партнёра`,
    c.`tin`                                      AS `ИНН`,
    s.`activated_keys`                           AS `Активированные ключи `,
    s.`activated_sub_w_key`                      AS `Вернувшиеся подписчики с ключем`,
    s.`created_first_sub_w_key`                  AS `Новые подписчики с ключем`,
    s.`d_activated_sub_w_key`                    AS `Отменившие подписку (вернувшиеся)`,
    s.`d_created_first_sub_w_key`                AS `Отменившие подписку (новые)`
FROM s
LEFT JOIN `t_a_city_uuid` AS e
    ON e.`city_uuid` = s.`city_uuid`
LEFT JOIN `companies_dir_partner_ch` AS c
    ON c.`partner_uuid` = s.`partner_uuid`
ORDER BY s.`report_date` DESC
LIMIT 100
```

### Пример суммарных метрик по диапазону дат
```sql
SELECT
    `report_date` AS `Дата`,
    SUM(`activated_keys`)               AS `Активированные ключи `,
    SUM(`activated_sub_w_key`)          AS `Вернувшиеся подписчики с ключем`,
    SUM(`created_first_sub_w_key`)      AS `Новые подписчики с ключем`,
    SUM(`d_activated_sub_w_key`)        AS `Отменившие подписку (вернувшиеся)`,
    SUM(`d_created_first_sub_w_key`)    AS `Отменившие подписку (новые)`
FROM `t_subscriptions_after_licensed_keys`
WHERE `report_date` BETWEEN toDate('2025-01-01') AND today()
GROUP BY `report_date`
ORDER BY `report_date` DESC
LIMIT 100
```
