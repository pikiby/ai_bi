---
id: t_ai_global_report__active_users
parent_id: t_ai_global_report
title: Активные пользователи (t_ai_global_report)
type: table-section
tags:
  - users
  - active_users
  - активные пользователи
  - city
  - partner
---

# Активные пользователи — t_ai_global_report

Кратко: дневные активные, новые активированные, BLE, срезы по городу/партнёру.

## Поля (выдержка)

| Поле                              | Тип    | Описание                                 |
| --------------------------------- | ------ | ---------------------------------------- |

## Примеры запросов

### ТОП-15 городов по активным пользователям
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
ORDER BY active_users DESC
LIMIT 15;
```

### Топ партнёров по paying_users (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
)
SELECT
  company_name,
  sum(paying_users) AS paying_users
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY  company_name
ORDER BY conversion DESC
LIMIT 20;
```


