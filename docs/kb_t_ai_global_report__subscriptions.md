---
id: t_ai_global_report__subscriptions
parent_id: t_ai_global_report
title: Подписки и коммерция (t_ai_global_report)
type: table-section
tags:
  - subscriptions
  - подписки
  - paying_users
  - partner
  - city
---

# Подписки и коммерция — t_ai_global_report

Кратко: метрики подписок Android/iOS, активированные подписки, PRO/стандартные, ошибки/отмены.

## Поля (выдержка)

| Поле                 | Тип    | Описание                                  |
| -------------------- | ------ | ----------------------------------------- |
| `paying_users`       | UInt64 | Активированные подписки (все)             |
| `paying_users_day`   | UInt64 | Активированные в день данных              |
| `android_sub`        | UInt64 | Подписки Android (историческое поле)      |
| `ios_sub`            | UInt64 | Подписки iOS через App Store              |
| `renew_failed_at`    | UInt64 | Неудачные продления/оплаты                |
| `renew_stopped_at`   | UInt64 | Отменённые подписки                       |

## Примеры запросов

### Топ партнёров по paying_users (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
)
SELECT
  company_name,
  sum(paying_users) AS paying
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY  company_name
ORDER BY paying DESC
LIMIT 20;
```

### Ошибки продлений за 28 дней по городам
```sql
SELECT
  city,
  sum(failed_subscript_last_28_days) AS failed_28d
FROM t_ai_global_report
WHERE report_date >= today() - 28
GROUP BY city
ORDER BY failed_28d DESC
LIMIT 15;
```


