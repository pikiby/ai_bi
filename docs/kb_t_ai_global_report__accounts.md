---
id: t_ai_global_report__accounts
parent_id: t_ai_global_report
title: Аккаунты и активации (t_ai_global_report)
type: table-section
tags:
  - accounts
  - активированные аккаунты
  - new_activated_account
---

# Аккаунты и активации — t_ai_global_report

Кратко: активированные аккаунты (всего/за день/за месяц), ИНН, ЛК партнёра.

## Поля (выдержка)

| Поле                         | Тип    | Описание                                  |
| ---------------------------- | ------ | ----------------------------------------- |
| `total_activated_account`    | UInt64 | Всего активированных аккаунтов            |
| `new_created_account_day`    | UInt64 | Новые созданные аккаунты за день          |
| `new_activated_account_day`  | UInt64 | Новые активированные аккаунты за день     |
| `tin`                        | String | ИНН партнёра                              |

## Примеры запросов

### Новые активированные аккаунты по городам (день)
```sql
SELECT
  city,
  sum(new_activated_account_day) AS new_activated_today
FROM t_ai_global_report
WHERE report_date = (SELECT max(report_date) FROM t_ai_global_report WHERE total_active_users > 0)
GROUP BY city
ORDER BY new_activated_today DESC
LIMIT 20;
```


