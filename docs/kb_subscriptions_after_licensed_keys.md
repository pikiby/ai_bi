---
id: t_subscriptions_after_licensed_keys
db: db1
short_description: "Дневной срез количества подписок, созданных после приобретения лицензионных ключей, по партнёрам и городам."
synonyms:
  - мобильные подписки
  - ключи
type: table
tags:
  - "#KnowledgeBase"
---

# «Подписки после приобретения ключей» — t_subscriptions_after_licensed_keys

## Назначение
Таблица фиксирует ежедневный срез по количеству подписок, созданных после приобретения лицензионных ключей, в разрезе партнёров и городов. Используется для анализа эффективности продаж лицензий и последующего вовлечения пользователей в подписки.

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
    `count_citid_sub_aft_entr` UInt32
)
ENGINE = MergeTree()
ORDER BY (report_date, partner_uuid, city_uuid)
```

## Поля и алиасы
| Поле                        | Тип    | Алиас (человекочитаемое)                |
|----------------------------|--------|-----------------------------------------|
| `report_date`              | Date   | Дата                                    |
| `partner_uuid`             | String | UUID партнёра                           |
| `city_uuid`                | String | UUID города                             |
| `count_citid_sub_aft_entr` | UInt32 | Количество подписок после лицензии      |


## Примечания
- В таблицу попадают только те подписки, которые были оформлены в течение 48 часов после приобретения лицензионного ключа.
- Для расчёта используются таблицы citizens_dir_mobile_ch, subscriptions_st_mobile_ch, licensed_keys_dir_partner_ch, entries_installation_points_dir_partner_ch, t_a_houses_by_partner.
- Используется для оценки влияния продаж лицензий на последующее подключение подписок.

## Связи и обогащение

- Город (человекочитаемо): из `entries_installation_points_dir_partner_ch`
  - Ключи связи: `partner_uuid`, `city_uuid`
  - Поле: `city` → алиас `Город`
- Реквизиты компании: из `companies_dir_partner`
  - Ключ связи: `partner_uuid`
  - Поля:
    - `company_name` → алиас `Компания`
    - `partner_lk` → алиас `ЛК партнёра`
    - `tin` → алиас `ИНН`

### Пример запроса (обогащённый, без префикса БД)
```sql
SELECT
    s.`report_date`                      AS `Дата`,
    s.`partner_uuid`                     AS `UUID партнёра`,
    s.`city_uuid`                        AS `UUID города`,
    e.`city`                             AS `Город`,
    c.`company_name`                     AS `Компания`,
    c.`partner_lk`                       AS `ЛК партнёра`,
    c.`tin`                              AS `ИНН`,
    s.`count_citid_sub_aft_entr`         AS `Подписок после лицензии`
FROM `t_subscriptions_after_licensed_keys` AS s
LEFT JOIN `entries_installation_points_dir_partner_ch` AS e
    ON e.`partner_uuid` = s.`partner_uuid`
   AND e.`city_uuid`    = s.`city_uuid`
LEFT JOIN `companies_dir_partner` AS c
    ON c.`partner_uuid` = s.`partner_uuid`
WHERE s.`report_date` = (
    SELECT max(`report_date`) FROM `t_subscriptions_after_licensed_keys`
)
ORDER BY s.`report_date` DESC
LIMIT 100
```
