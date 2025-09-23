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

## Поля и связи (CH)
- `partner_uuid` — связь с партнёром
- `city_uuid` — связь с городом
- `city` — человекочитаемое название города (алиас: `Город`)

## Пример (CH)
```sql
SELECT `partner_uuid`, `city_uuid`, `city`
FROM `entries_installation_points_dir_partner_ch`
LIMIT 10
```


