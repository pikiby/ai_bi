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

## Движок и сортировка
- Источник: объект S3 (CSVWithNames)
- Проекция в CH: `companies_dir_partner_ch` (MergeTree, ORDER BY `partner_uuid`)

## DDL (референс)
```sql
-- Источник
CREATE TABLE IF NOT EXISTS companies_dir_partner
(
    `company_name` String,
    `partner_lk` String,
    `registration_date` Date,
    `partner_uuid` String,
    `tin` String,
    `kpp` String
)
ENGINE = S3('...','CSVWithNames')

-- Проекция в CH
CREATE TABLE IF NOT EXISTS companies_dir_partner_ch
(
    `company_name` String,
    `partner_lk` String,
    `registration_date` String,
    `partner_uuid` String,
    `tin` String,
    `kpp` String
)
ENGINE = MergeTree()
ORDER BY partner_uuid
```

## Ключевые поля и связи
- `partner_uuid` — ключ связи с фактами/витринами
- Рекомендуемые алиасы:
  - `company_name` → `Компания`
  - `partner_lk` → `ЛК партнёра`
  - `tin` → `ИНН`

## Пример запроса
```sql
SELECT
    `partner_uuid`,
    any(`company_name`) AS `Компания`,
    any(`partner_lk`)   AS `ЛК партнёра`,
    any(`tin`)          AS `ИНН`
FROM `companies_dir_partner`
GROUP BY `partner_uuid`
LIMIT 100
```


