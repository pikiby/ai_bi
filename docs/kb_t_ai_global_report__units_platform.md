---
id: t_ai_global_report__units
parent_id: t_ai_global_report
title: Единицы на платформе (t_ai_global_report)
type: table-section
tags:
  - units
  - platform
  - единицы
  - монетизация
---

# Единицы на платформе — t_ai_global_report

Кратко: свободная/ограниченная монетизация, суммарные показатели по единицам.

## Поля (выдержка)

| Поле                        | Тип    | Описание                      |
| --------------------------- | ------ | ----------------------------- |
| `units_free_monetization`   | UInt64 | Единицы со свободной монетизацией |
| `units_free_monetization_pro` | UInt64 | Единицы PRO                 |
| `units_on_platform`         | UInt64 | Общее количество единиц      |

## Примеры запросов

### ТОП партнёров по количеству единиц (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  partner_lk,
  sum(units_on_platform) AS units
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY partner_lk
ORDER BY units DESC
LIMIT 20;
```


