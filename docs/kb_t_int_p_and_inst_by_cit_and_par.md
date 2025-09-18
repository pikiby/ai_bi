---
id: table_t_int_p_and_inst_by_cit_and_par
db: db1
short_description: "Дневной срез точек установки (адресов) и домофонов по городам и компаниям/партнёрам: всего и онлайн."
synonyms:
  - точки установки
  - домофоны
  - компании
  - партнёры
  - города
type: table
source:
  - "[[t_int_p_and_inst_by_cit_and_par]]"
tags:
  - "#KnowledgeBase"
---

# «Точки установки и домофоны по городам и компаниям» — t_int_p_and_inst_by_cit_and_par

## Назначение
Таблица фиксирует ежедневный срез по точкам установки (адресам) и домофонам в разрезе городов и компаний/партнёров. Используется для сводных отчётов «Домофоны по городам и компаниям», мониторинга онлайна и аналитики проникновения.

## Хранилище и движок
- БД: `db1`
- Таблица: `t_int_p_and_inst_by_cit_and_par`
- Движок: `MergeTree`
- Сортировка: `ORDER BY report_date`

## DDL
```sql
CREATE TABLE t_int_p_and_inst_by_cit_and_par
(
    `report_date` Date,
    `city` String,
    `city_uuid` String,
    `company_name` String,
    `partner_uuid` String,
    `tin` String,
    `partner_lk` String,
    `countd_inst_by_c_and_p` UInt64,
    `countd_inter_par_by_c_and_p` UInt64,
    `countd_inst_par_onl_by_c_and_p` UInt64,
    `countd_inter_par_onl_by_c_and_p` UInt64
)
ENGINE = MergeTree()
ORDER BY report_date;
```

## Поля и алиасы
| Поле                                   | Тип    | Алиас (человекочитаемое)                                  |
|----------------------------------------|--------|-----------------------------------------------------------|
| `report_date`                          | Date   | Дата                                                      |
| `city`                                 | String | Город                                                     |
| `city_uuid`                            | String | UUID города                                               |
| `company_name`                         | String | Название компании (партнёра)                              |
| `partner_uuid`                         | String | UUID компании/партнёра                                    |
| `tin`                                  | String | ИНН                                                       |
| `partner_lk`                           | String | Личный кабинет                                            |
| `countd_inst_by_c_and_p`               | UInt64 | Количество точек установки                                |
| `countd_inter_par_by_c_and_p`          | UInt64 | Количество точек установки с домофоном онлайн             |
| `countd_inst_par_onl_by_c_and_p`       | UInt64 | Количество домофонов                                      |
| `countd_inter_par_onl_by_c_and_p`      | UInt64 | Количество домофонов онлайн                               |

## Примечания
- **Партнёр = компания.**
- **Точка установки** — адрес, на котором числится домофон.
- На одной точке установки может быть **несколько домофонов**.

## Частые срезы/фильтры
- По дате (`report_date`) — чаще всего последняя доступная дата.
- По городу (`city`, `city_uuid`).
- По компании/партнёру (`company_name`, `partner_uuid`, `tin`, `partner_lk`).

## Примеры запросов (ClickHouse)

1) **Сводка по городам и компаниям за последнюю дату**
```sql
WITH last_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_int_p_and_inst_by_cit_and_par
)
SELECT
  report_date,
  city            AS "город",
  company_name    AS "компания",
  partner_lk      AS "личный_кабинет",
  tin             AS "ИНН",
  countd_inst_by_c_and_p        AS "точек_установки",
  countd_inter_par_by_c_and_p   AS "точек_онлайн",
  countd_inst_par_onl_by_c_and_p  AS "домофонов",
  countd_inter_par_onl_by_c_and_p AS "домофонов_онлайн"
FROM t_int_p_and_inst_by_cit_and_par t
INNER JOIN last_dt USING (report_date)
ORDER BY countd_inst_by_c_and_p DESC;
```

2) **Онлайн-доля по компаниям в городе**
```sql
WITH last_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_int_p_and_inst_by_cit_and_par
)
SELECT
  report_date,
  company_name                           AS `компания`,
  countd_inst_par_onl_by_c_and_p         AS `домофонов`,
  countd_inter_par_onl_by_c_and_p        AS `домофонов_онлайн`,
  countd_inter_par_onl_by_c_and_p / nullIf(countd_inst_par_onl_by_c_and_p, 0) AS `доля_онлайн`
FROM t_int_p_and_inst_by_cit_and_par
INNER JOIN last_dt USING (report_date)
WHERE city = '<<ГОРОД>>'
ORDER BY `доля_онлайн` DESC;
```

3) **Динамика по компании**
```sql
SELECT
  report_date,
  countd_inst_by_c_and_p           AS points,
  countd_inst_par_onl_by_c_and_p   AS intercoms,
  countd_inter_par_onl_by_c_and_p  AS intercoms_online,
  countd_inter_par_onl_by_c_and_p / nullIf(countd_inst_par_onl_by_c_and_p, 0) AS online_rate
FROM t_int_p_and_inst_by_cit_and_par
WHERE partner_uuid = '<<company_name>>'
ORDER BY report_date;
```

## Возможные связи
- Со справочником городов — по `city_uuid`.
- Со справочником партнёров — по `partner_uuid`/`tin`.
- С адресным справочником (точки/подъезды) — по внешнему ключу, если предусмотрен.
