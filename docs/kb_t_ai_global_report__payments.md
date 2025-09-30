---
id: t_ai_global_report__payments
parent_id: t_ai_global_report
title: Платежи и доходы (t_ai_global_report)
type: table-section
tags:
  - payments
  - revenue
  - оплаты
  - платежи
  - Android_PL
  - IOS_PL
  - city
  - partner
---

# Платежи и доходы — t_ai_global_report

Кратко: дневные и кумулятивные платежи по Android/iOS, покупки по тарифам, возвраты.

## Поля

| Поле                | Тип     | Описание                         |
| ------------------- | ------- | -------------------------------- |
| `Android_PL`        | Float64 | Дневная выручка Android          |
| `IOS_PL`            | Float64 | Дневная выручка iOS              |
| `refunded_amount_appstore` | Float64 | Возвраты App Store (день)      |
| `refunded_amount_yookassa` | Float64 | Возвраты YooKassa (день)       |

Кумулятивные метрики за месяц: `Android_PL_cum`, `IOS_PL_cum`, счетчики покупок *_cum.

## Частые задачи

### Топ-10 городов по дневной выручке
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  sum(Android_PL + IOS_PL) AS revenue
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY revenue DESC
LIMIT 10;
```

### Топ партнеров по месячной выручке
```sql
WITH month_bounds AS (
  SELECT toStartOfMonth(today()) AS m_start, 
         addMonths(toStartOfMonth(today()), 1) AS m_end)
SELECT
  partner_lk,
  sum(IOS_PL + Android_PL) AS total_monthly_revenue
FROM t_ai_global_report
WHERE report_date >= m_start AND report_date < m_end
GROUP BY partner_lk
ORDER BY total_monthly_revenue DESC
LIMIT 15;
```

### Возвраты по платформам (последняя дата)
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  sum(refunded_amount_appstore) AS appstore_refunds,
  sum(refunded_amount_yookassa) AS yookassa_refunds
FROM t_ai_global_report
INNER JOIN last_date USING (report_date);
```


