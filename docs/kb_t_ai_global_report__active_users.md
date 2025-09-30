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
| `total_active_users_per_day`      | UInt64 | Активные пользователи за день            |
| `total_active_users`              | UInt64 | Всего активных пользователей             |
| `new_active_users`                | UInt64 | Новые активированные в текущем месяце    |
| `total_active_users_ble_available`| UInt64 | Активные с BLE                           |

## Примеры запросов

### ТОП-15 городов по активным пользователям
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  sum(total_active_users) AS active_users
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY active_users DESC
LIMIT 15;
```

### Конверсия paying_users/total_active_users по партнёрам (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  company_name,
  sum(paying_users) / nullIf(sum(total_active_users), 0) AS conversion
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY  company_name
ORDER BY conversion DESC
LIMIT 20;
```


