---
id: companies_dir_partner
db: db1
short_description: "Справочник партнёрских компаний и их реквизитов."
synonyms:
  - компании партнёров
  - реквизиты партнёров
  - партнёрские компании
type: table
tags:
  - "#KnowledgeBase"
---

# `companies_dir_partner`

## Назначение
Справочник компаний партнёров с ключом `partner_uuid` и реквизитами: название компании, ЛК партнёра, ИНН и т.д. Используется для обогащения витрин партнёрскими атрибутами.

## Поля и связи (CH)
- Таблица для обогащения: `companies_dir_partner_ch`
- Ключ: `partner_uuid`
- Рекомендуемые алиасы:
  - `company_name` → `Компания`
  - `partner_lk` → `ЛК партнёра`
  - `tin` → `ИНН`

## Пример (CH)
```sql
SELECT `partner_uuid`, `company_name`, `partner_lk`, `tin`
FROM `companies_dir_partner_ch`
LIMIT 10
```


