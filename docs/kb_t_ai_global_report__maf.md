---
id: t_ai_global_report__maf
parent_id: t_ai_global_report
title: MAF — Monthly Active Flats (t_ai_global_report)
type: table-section
tags:
  - maf
  - квартиры
  - активность
---

# MAF — Monthly Active Flats

Кратко: активность на уровне квартир, срезы по городам/партнёрам.

## Поля (выдержка)

| Поле        | Тип    | Описание                                  |
| ----------- | ------ | ----------------------------------------- |
| `MAF`       | UInt64 | Уникальные квартиры с активностью за день |
| `stricted_MAF` | UInt64 | Квартиры с ограниченной монетизацией  |
| `freemonetization_MAF` | UInt64 | Квартиры со свободной монетизацией |

## Примеры запросов

### ТОП-20 городов по MAF (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  sum(MAF) AS maf
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY maf DESC
LIMIT 20;
```


