---
id: table_t_billing_and_accruals
db: db1
short_description: "Ежедневные данные по биллингу и начислениям партнёров: платежи, подписки, статусы компаний."
synonyms:
  - биллинг
  - начисления
  - партнёры
  - платежи
  - подписки
  - компании
type: table
source:
  - "[[t_billing_and_accruals]]"
tags:
  - "#KnowledgeBase"
---

# «Биллинг и начисления партнёров» — t_billing_and_accruals

## Назначение
Таблица содержит ежедневные данные по биллингу и начислениям партнёров, включая информацию о платежах, подписках, статусах компаний и программе партнёрства. Используется для аналитики доходов, мониторинга подписок и расчёта партнёрских начислений.

## Хранилище и движок
- БД: `db1`
- Таблица: `t_billing_and_accruals`
- Движок: `MergeTree`
- Сортировка: `ORDER BY report_date`

## DDL
```sql
CREATE TABLE t_billing_and_accruals
(
    `report_date` Date,
    `report_date_month` Date,
    `report_date_week` Date,
    `report_date_year` Date,
    `company_name` String,
    `partner_uuid` String,
    `partner_lk` String,
    `pro_subs` UInt8,
    `tin` String,
    `billing_account_id` String,
    `kpp` String,
    `service` String,
    `billing_cost` Float64,
    `partner_program_accruals_amount` Float64,
    `billing_sum` Float64,
    `partner_in_lk_list` UInt8,
    `status` String,
    `billing_pro` UInt8,
    `enterprise_not_paid` UInt8,
    `enterprise_subs` UInt8,
    `enterprise_test` UInt8,
    `is_blocked` UInt8,
    `kz_pro` UInt8,
    `billing_count_new` Float64
)
ENGINE = MergeTree()
ORDER BY report_date;
```

## Поля и алиасы
| Поле                                   | Тип      | Алиас (человекочитаемое)                                  |
|----------------------------------------|----------|-----------------------------------------------------------|
| `report_date`                          | Date     | Дата отчёта                                               |
| `report_date_month`                    | Date     | Начало месяца                                             |
| `report_date_week`                     | Date     | Начало недели                                             |
| `report_date_year`                     | Date     | Начало года                                               |
| `company_name`                         | String   | Название компании (партнёра)                              |
| `partner_uuid`                         | String   | UUID компании/партнёра                                    |
| `partner_lk`                           | String   | Личный кабинет                                            |
| `pro_subs`                             | UInt8    | PRO подписка                                              |
| `tin`                                  | String   | ИНН                                                       |
| `billing_account_id`                   | String   | ID биллингового аккаунта                                  |
| `kpp`                                  | String   | КПП                                                       |
| `service`                              | String   | Услуга                                                    |
| `billing_cost`                         | Float64  | Стоимость биллинга                                        |
| `partner_program_accruals_amount`      | Float64  | Сумма начислений по партнёрской программе                 |
| `billing_sum`                          | Float64  | Сумма биллинга                                            |
| `partner_in_lk_list`                   | UInt8    | Партнёр в списке ЛК                                        |
| `status`                               | String   | Статус (Start/PRO/Enterprise)                             |
| `billing_pro`                          | UInt8    | Биллинг PRO                                               |
| `enterprise_not_paid`                  | UInt8    | Enterprise не оплачен                                     |
| `enterprise_subs`                      | UInt8    | Enterprise подписка                                       |
| `enterprise_test`                      | UInt8    | Enterprise тест                                           |
| `is_blocked`                           | UInt8    | Заблокирован                                              |
| `kz_pro`                               | UInt8    | КЗ PRO                                                    |
| `billing_count_new`                    | Float64  | Новое количество биллинга                                 |

## Примечания
- **Партнёр = компания.**
- **Статус** определяется автоматически: Enterprise (если enterprise_subs=1 или enterprise_not_paid=1), PRO (если pro_subs=1), иначе Start.
- **partner_in_lk_list** - флаг для партнёров из специального списка ЛК.
- Исключены тестовые и служебные партнёры из анализа.

## Частые срезы/фильтры
- По дате (`report_date`, `report_date_month`, `report_date_week`, `report_date_year`).
- По компании/партнёру (`company_name`, `partner_uuid`, `partner_lk`, `tin`).
- По статусу (`status`).
- По типу подписки (`pro_subs`, `enterprise_subs`).
- По блокировке (`is_blocked`).

## Примеры запросов (ClickHouse)

1) **Сводка по компаниям за последнюю дату**
```sql
WITH last_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_billing_and_accruals
)
SELECT
  report_date,
  company_name    AS "компания",
  partner_lk      AS "личный_кабинет",
  tin             AS "ИНН",
  status          AS "статус",
  billing_sum     AS "сумма_биллинга",
  partner_program_accruals_amount AS "начисления_партнёрской_программы",
  is_blocked      AS "заблокирован"
FROM t_billing_and_accruals t
INNER JOIN last_dt USING (report_date)
ORDER BY billing_sum DESC;
```

2) **Месячная динамика по статусам**
```sql
SELECT
  report_date_month AS "месяц",
  status            AS "статус",
  count()           AS "количество_компаний",
  sum(billing_sum)  AS "общая_сумма_биллинга",
  avg(billing_sum)  AS "средняя_сумма_биллинга"
FROM t_billing_and_accruals
WHERE report_date >= toDate('2024-01-01')
GROUP BY report_date_month, status
ORDER BY report_date_month, status;
```

3) **Топ-10 компаний по начислениям партнёрской программы**
```sql
WITH last_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_billing_and_accruals
)
SELECT
  company_name                        AS "компания",
  partner_lk                          AS "личный_кабинет",
  status                              AS "статус",
  partner_program_accruals_amount     AS "начисления_партнёрской_программы",
  billing_sum                         AS "сумма_биллинга"
FROM t_billing_and_accruals t
INNER JOIN last_dt USING (report_date)
WHERE partner_program_accruals_amount > 0
ORDER BY partner_program_accruals_amount DESC
LIMIT 10;
```

4) **Анализ заблокированных компаний**
```sql
SELECT
  status                              AS "статус",
  count()                             AS "всего_компаний",
  sum(is_blocked)                     AS "заблокированных",
  sum(is_blocked) / count() * 100     AS "процент_заблокированных"
FROM t_billing_and_accruals
WHERE report_date = (SELECT max(report_date) FROM t_billing_and_accruals)
GROUP BY status
ORDER BY status;
```

## Возможные связи
- Со справочником партнёров — по `partner_uuid`/`tin`.
- С таблицей биллинговых заказов — по `billing_account_id`.
- С таблицей начислений — по `partner_uuid` и `report_date`.
