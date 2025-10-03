---
id: t_ai_global_report__maf_monthly
parent_id: t_ai_global_report
title: Месячные метрики активности (t_ai_global_report)
type: table-section
tags:
  - maf
  - monthly
  - квартиры
  - активность
  - месяц
---

# Месячные метрики активности — t_ai_global_report

Кратко: агрегированные показатели активности квартир за месяц с разбивкой по типам монетизации и доступности ключей.

## Поля (выдержка)

| Поле                        | Тип    | Описание                      |
| --------------------------- | ------ | ----------------------------- |

## Примеры запросов

### ТОП-20 городов по месячным метрикам активности (последняя дата)
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
ORDER BY maf_monthly DESC
LIMIT 20;
```

### Анализ доступности ключей по партнерам
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
)
SELECT
  company_name,
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY company_name
HAVING total_stricted > 0 OR total_freemonetization > 0
ORDER BY (stricted_with_keys + freemonetization_with_keys) DESC
LIMIT 15;
```

### Сравнение дневных и месячных метрик активности
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
ORDER BY maf_monthly DESC
LIMIT 10;
```
