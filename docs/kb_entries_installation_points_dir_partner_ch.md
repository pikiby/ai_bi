---
id: entries_installation_points_dir_partner_ch
db: db1
short_description: "Каталог точек установки (подъезды) с привязкой к партнёрам и городам."
synonyms:
  - точки установки
  - подъезды
  - адреса партнёров
type: table
tags:
  - "#KnowledgeBase"
---

# `entries_installation_points_dir_partner_ch`

## Назначение
Справочник точек установки у партнёров (подъезды/адреса) с полями города и UUID города. Используется для обогащения витрин городской метрикой/названием.

## Движок и сортировка
- Движок: `MergeTree`
- Сортировка: `ORDER BY installation_point_id`

## DDL (референс)
```sql
CREATE TABLE IF NOT EXISTS entries_installation_points_dir_partner_ch
(
    `full_address` String,
    `created_at` String,
    `number` Int32,
    `lat` String,
    `lon` String,
    `first_flat` Int16,
    `last_flat` Int16,
    `flats_count_full` Int16,
    `flats_count` Int16,
    `address_uuid` String,
    `parent_uuid` String,
    `partner_uuid` String,
    `installation_point_id` Int64,
    `region` String,
    `country` String,
    `city` String,
    `city_uuid` String,
    `parts` Array(String),
    `entrance_number` String,
    `building_number` String,
    `streat_name` String
)
ENGINE = MergeTree()
ORDER BY installation_point_id
```

## Ключевые поля и связи
- `partner_uuid` — связь с партнёром
- `city_uuid` — связь с городом
- `city` — человекочитаемое название города (алиас: `Город`)

## Пример запроса
```sql
SELECT
    `partner_uuid`,
    `city_uuid`,
    `city` AS `Город`,
    count() AS `Кол-во точек`
FROM `entries_installation_points_dir_partner_ch`
GROUP BY `partner_uuid`, `city_uuid`, `city`
ORDER BY `Кол-во точек` DESC
LIMIT 50
```


