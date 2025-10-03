---
id: t_ai_global_report__maf
parent_id: t_ai_global_report
title: Метрики активности (t_ai_global_report)
type: table-section
tags:
  - maf
  - квартиры
  - активность
---

# Метрики активности

Кратко: активность на уровне квартир, срезы по городам/партнёрам.

## Поля (выдержка)

| Поле        | Тип    | Описание                                  |
| ----------- | ------ | ----------------------------------------- |

## Примеры запросов

### ТОП-20 городов по активности (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
)
SELECT
  city,
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY maf DESC
LIMIT 20;
```


