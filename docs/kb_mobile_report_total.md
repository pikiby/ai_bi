---
id: t_mobile_report_total
title: Мобильный отчет по подпискам и платежам
db: db1
short_description: "Ежедневный отчет по мобильным подпискам, платежам, активным пользователям и единицам на платформе с разбивкой по партнерам и городам."
synonyms:
  - мобильный отчет
  - отчет по подпискам
  - отчет по платежам
  - мобильная аналитика
  - подписки и платежи
type: table
tags:
  - "#KnowledgeBase"
  - "#Mobile_Report"
---

# «Мобильный отчет по подпискам и платежам» — mobile_report_total

## Названия таблицы
**Короткое имя (человекочитаемое):** Мобильный отчет по подпискам и платежам  
**Тех. имя:** `mobile_report_total`

## Назначение
Таблица предоставляет ежедневный срез по мобильным подпискам, платежам, активным пользователям и единицам на платформе с разбивкой по партнерам и городам. Используется для аналитики мобильного бизнеса, мониторинга доходов, анализа пользовательской активности и контроля качества данных.

**Основные метрики:**
- Подписки (Android/iOS, новые/продленные, через App Store/карты)
- Платежи (доходы, покупки, возвраты по платформам и тарифам)
- Активные пользователи (общие, с монетизацией, с BLE)
- Единицы на платформе (свободная/ограниченная монетизация)

## Хранилище и движок
- БД: `db1`  
- Таблица: `mobile_report_total`  
- Движок: `MergeTree`  
- Сортировка: `ORDER BY report_date`
- Материализованное представление: `mobile_report_total_mv` (обновляется ежедневно в 06:05)

## DDL
```sql
CREATE MATERIALIZED VIEW db1.mobile_report_total_mv
    REFRESH EVERY 1 DAY OFFSET 6 HOUR 5 MINUTE TO db1.mobile_report_total AS
    WITH t_dec AS(
        SELECT
            t1.report_date  AS report_date,
            t2.city  AS city,
            t2.partner_uuid  AS partner_uuid
        FROM
            (SELECT DISTINCT report_date FROM db1.installation_point_st_partner_ch) AS t1
        CROSS JOIN
            (SELECT DISTINCT city,inst.partner_uuid AS partner_uuid
            FROM db1.installation_point_st_partner_ch AS inst
            LEFT JOIN db1.entries_installation_points_dir_partner AS eipdp ON inst.installation_point_id  = eipdp.installation_point_id ) AS t2
        )
    SELECT
        t_dec.report_date AS report_date,
        t_dec.partner_uuid AS partner_uuid,
        t_dec.city AS city,
        -- Метрики подписок, платежей, пользователей и единиц
        -- (полный список полей см. в разделе "Поля и алиасы")
    FROM t_dec
    LEFT JOIN db1.units_on_sk_platform_rep_mobile_total AS uosprmt 
        ON uosprmt.report_date = t_dec.report_date 
        AND uosprmt.city = t_dec.city
        AND uosprmt.partner_uuid = t_dec.partner_uuid
    -- Другие JOIN'ы с источниками данных
    ORDER BY report_date DESC;
```

## Поля и алиасы

### Основные поля
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `report_date`                 | Date    | Дата отчета                               |
| `partner_uuid`                | String  | UUID партнера                             |
| `city`                        | String  | Город                                     |

### Метрики подписок
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `android_sub`                 | UInt64  | Подписки Android (старые данные)         |
| `android_sub_extended_new`    | UInt64  | Продленные подписки Android (старые данные) |
| `android_sub_extended_new_cart` | UInt64 | Продленные подписки Android с карты      |
| `android_sub_first_new`      | UInt64  | Новые подписки Android (старые данные)    |
| `android_sub_first_new_cart` | UInt64  | Новые подписки Android с карты            |
| `android_sub_from_cart`       | UInt64  | Подписки Android с карты                  |
| `ios_sub`                    | UInt64  | Подписки iOS через App Store              |
| `ios_sub_extended_new`       | UInt64  | Продленные подписки iOS через App Store   |
| `ios_sub_extended_new_cart`  | UInt64  | Продленные подписки iOS с карты           |
| `ios_sub_first_new`          | UInt64  | Новые подписки iOS через App Store        |
| `ios_sub_first_new_cart`     | UInt64  | Новые подписки iOS с карты                |
| `ios_sub_first_new_cart_transition` | UInt64 | Новые подписки iOS с карты (переход из App Store) |
| `ios_sub_from_cart`          | UInt64  | Подписки iOS с карты                      |
| `paying_users`               | UInt64  | Все активированные подписки               |
| `paying_users_day`           | UInt64  | Подписки, активированные в день отчета    |
| `paying_users_partner_pro`   | UInt64  | PRO подписки партнеров                    |
| `paying_users_standart`      | UInt64  | Стандартные подписки                      |
| `paying_users_standart_appstore` | UInt64 | Подписки iOS через App Store             |
| `paying_users_standart_ios_from_cart` | UInt64 | Подписки iOS с карты (переход с App Store) |
| `paying_users_standart_yakassa` | UInt64 | Подписки Android через YooKassa (старые данные) |

### Метрики неудачных операций
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `failed_subscript_last_28_days` | UInt64 | Ошибки оплаты за последние 28 дней        |
| `renew_failed_at`            | UInt64  | Общее количество неудачных продлений/оплат |
| `renew_failed_at_month`     | UInt64  | Неудачные продления/оплаты в текущем месяце (кумулятивно) |
| `renew_failed_at_andeoid_cart` | UInt64 | Неудачные продления Android с карты       |
| `renew_failed_at_android`   | UInt64  | Неудачные продления Android (старые данные) |
| `renew_failed_at_ios`       | UInt64  | Неудачные продления iOS через App Store   |
| `renew_failed_at_ios_cart`  | UInt64  | Неудачные продления iOS с карты           |
| `renew_stopped_at`           | UInt64  | Общее количество отмененных подписок      |
| `renew_stopped_at_month`    | UInt64  | Отмены подписок в текущем месяце (кумулятивно) |
| `renew_stopped_at_android`  | UInt64  | Отмененные подписки Android (старые данные) |
| `renew_stopped_at_android_cart` | UInt64 | Отмененные подписки Android с карты       |
| `renew_stopped_at_ios`      | UInt64  | Отмененные подписки iOS через App Store   |
| `renew_stopped_at_ios_cart` | UInt64  | Отмененные подписки iOS с карты           |

### Метрики единиц на платформе
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `units_free_monetization_start` | UInt64 | Единицы со стартовой PRO подпиской (PRO без enterprise) |
| `units_free_monetization`    | UInt64  | Единицы со свободной монетизацией         |
| `units_free_monetization_pro` | UInt64 | Единицы со свободной монетизацией PRO     |
| `units_on_platform`          | UInt64  | Общее количество единиц на платформе     |
| `units_stricted monetization` | UInt64 | Единицы с ограниченной монетизацией       |

### Метрики активных пользователей
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `total_active_user_monetization_per_day` | UInt64 | Активные пользователи с монетизацией за день |
| `total_active_user_subscribed_monetization_per_day` | UInt64 | Активные пользователи с подпиской и монетизацией за день |
| `total_active_users_per_day` | UInt64  | Общее количество активных пользователей за день |
| `new_active_users`           | UInt64  | Новые пользователи (активированы в текущем месяце) |
| `total_active_user_subscribed_monetization` | UInt64 | Активные пользователи с подпиской и монетизацией |
| `total_active_users`         | UInt64  | Общее количество активных пользователей  |
| `total_active_users_ble_available` | UInt64 | Активные пользователи с доступным BLE     |
| `total_active_users_ble_available_monetization` | UInt64 | Активные пользователи с BLE и монетизацией |
| `total_active_users_ble_available_subscribed_monetization` | UInt64 | Активные пользователи с BLE, монетизацией и подпиской |
| `total_active_users_monetization` | UInt64 | Активные пользователи с монетизацией      |

### Метрики платежей (дневные)
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `Android_PL`                 | Float64 | Прибыль от Android за день                |
| `IOS_PL`                     | Float64 | Прибыль от iOS за день                     |
| `appstore_count_1`           | UInt64  | Покупки за 1₽ в App Store за день         |
| `appstore_count_1_refunded`  | UInt64  | Возвраты за 1₽ в App Store за день        |
| `appstore_count_2390`        | UInt64  | Покупки за 2390₽ в App Store за день      |
| `appstore_count_2390_refunded` | UInt64 | Возвраты за 2390₽ в App Store за день     |
| `appstore_count_499`         | UInt64  | Покупки за 499₽ в App Store за день       |
| `appstore_count_499_refunded` | UInt64 | Возвраты за 499₽ в App Store за день      |
| `appstore_count_69`          | UInt64  | Покупки за 69₽ в App Store за день        |
| `appstore_count_69_refunded` | UInt64  | Возвраты за 69₽ в App Store за день       |
| `appstore_count_85`          | UInt64  | Покупки за 85₽ в App Store за день        |
| `appstore_count_85_refunded` | UInt64  | Возвраты за 85₽ в App Store за день       |
| `yookassa_count_1`           | UInt64  | Покупки за 1₽ через YooKassa за день      |
| `yookassa_count_1_refunded`  | UInt64  | Возвраты за 1₽ через YooKassa за день     |
| `yookassa_count_2390`        | UInt64  | Покупки за 2390₽ через YooKassa за день  |
| `yookassa_count_2390_refunded` | UInt64 | Возвраты за 2390₽ через YooKassa за день |
| `yookassa_count_249`         | UInt64  | Покупки за 249₽ через YooKassa за день    |
| `yookassa_count_249_refunded` | UInt64 | Возвраты за 249₽ через YooKassa за день  |
| `yookassa_count_35`          | UInt64  | Покупки за 35₽ через YooKassa за день     |
| `yookassa_count_35_refunded` | UInt64  | Возвраты за 35₽ через YooKassa за день   |
| `yookassa_count_499`         | UInt64  | Покупки за 499₽ через YooKassa за день    |
| `yookassa_count_499_refunded` | UInt64 | Возвраты за 499₽ через YooKassa за день  |
| `yookassa_count_69`          | UInt64  | Покупки за 69₽ через YooKassa за день    |
| `yookassa_count_69_refunded` | UInt64  | Возвраты за 69₽ через YooKassa за день   |
| `yookassa_count_85`          | UInt64  | Покупки за 85₽ через YooKassa за день    |
| `yookassa_count_85_refunded` | UInt64  | Возвраты за 85₽ через YooKassa за день   |
| `refunded_amount_appstore`   | Float64 | Сумма возвратов в App Store за день       |
| `refunded_amount_yookassa`   | Float64 | Сумма возвратов через YooKassa за день    |

### Кумулятивные метрики платежей (месячные)
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `Android_PL_cum`             | Float64 | Кумулятивная прибыль Android за месяц     |
| `IOS_PL_cum`                 | Float64 | Кумулятивная прибыль iOS за месяц         |
| `appstore_count_1_cum`       | UInt64  | Кумулятивные покупки за 1₽ в App Store за месяц |
| `appstore_count_1_refunded_cum` | UInt64 | Кумулятивные возвраты за 1₽ в App Store за месяц |
| `appstore_count_2390_cum`     | UInt64  | Кумулятивные покупки за 2390₽ в App Store за месяц |
| `appstore_count_2390_refunded_cum` | UInt64 | Кумулятивные возвраты за 2390₽ в App Store за месяц |
| `appstore_count_499_cum`      | UInt64  | Кумулятивные покупки за 499₽ в App Store за месяц |
| `appstore_count_499_refunded_cum` | UInt64 | Кумулятивные возвраты за 499₽ в App Store за месяц |
| `appstore_count_69_cum`       | UInt64  | Кумулятивные покупки за 69₽ в App Store за месяц |
| `appstore_count_69_refunded_cum` | UInt64 | Кумулятивные возвраты за 69₽ в App Store за месяц |
| `appstore_count_85_cum`       | UInt64  | Кумулятивные покупки за 85₽ в App Store за месяц |
| `appstore_count_85_refunded_cum` | UInt64 | Кумулятивные возвраты за 85₽ в App Store за месяц |
| `yookassa_count_1_cum`        | UInt64  | Кумулятивные покупки за 1₽ через YooKassa за месяц |
| `yookassa_count_1_refunded_cum` | UInt64 | Кумулятивные возвраты за 1₽ через YooKassa за месяц |
| `yookassa_count_2390_cum`     | UInt64  | Кумулятивные покупки за 2390₽ через YooKassa за месяц |
| `yookassa_count_2390_refunded_cum` | UInt64 | Кумулятивные возвраты за 2390₽ через YooKassa за месяц |
| `yookassa_count_249_cum`      | UInt64  | Кумулятивные покупки за 249₽ через YooKassa за месяц |
| `yookassa_count_249_refunded_cum` | UInt64 | Кумулятивные возвраты за 249₽ через YooKassa за месяц |
| `yookassa_count_35_cum`       | UInt64  | Кумулятивные покупки за 35₽ через YooKassa за месяц |
| `yookassa_count_35_refunded_cum` | UInt64 | Кумулятивные возвраты за 35₽ через YooKassa за месяц |
| `yookassa_count_499_cum`      | UInt64  | Кумулятивные покупки за 499₽ через YooKassa за месяц |
| `yookassa_count_499_refunded_cum` | UInt64 | Кумулятивные возвраты за 499₽ через YooKassa за месяц |
| `yookassa_count_69_cum`       | UInt64  | Кумулятивные покупки за 69₽ через YooKassa за месяц |
| `yookassa_count_69_refunded_cum` | UInt64 | Кумулятивные возвраты за 69₽ через YooKassa за месяц |
| `yookassa_count_85_cum`       | UInt64  | Кумулятивные покупки за 85₽ через YooKassa за месяц |
| `yookassa_count_85_refunded_cum` | UInt64 | Кумулятивные возвраты за 85₽ через YooKassa за месяц |
| `refunded_amount_appstore_cum` | Float64 | Кумулятивная сумма возвратов в App Store за месяц |
| `refunded_amount_yookassa_cum` | Float64 | Кумулятивная сумма возвратов через YooKassa за месяц |

### Метрики активированных аккаунтов
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `total_activated_account`     | UInt64  | Общее количество активированных аккаунтов |
| `total_activated_account_monetization` | UInt64 | Активированные аккаунты с монетизацией |
| `total_activated_account_ble_available_monetization` | UInt64 | Активированные аккаунты с BLE и монетизацией |
| `total_activated_account_ble_available` | UInt64 | Активированные аккаунты с доступным BLE |
| `new_created_account`         | UInt64  | Кумулятивное количество новых созданных аккаунтов за месяц |
| `new_activated_account`       | UInt64  | Кумулятивное количество новых активированных аккаунтов за месяц |
| `new_created_account_day`     | UInt64  | Новые созданные аккаунты за день          |
| `new_activated_account_day`   | UInt64  | Новые активированные аккаунты за день     |

### Метрики MAF (Monthly Active Flats)
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `MAF`                         | UInt64  | Уникальные квартиры с активными пользователями за день |
| `stricted_MAF`               | UInt64  | Квартиры с ограниченной монетизацией      |
| `freemonetization_MAF`       | UInt64  | Квартиры со свободной монетизацией        |

## Примечания к полям

### Типы метрик
- **Дневные метрики** — значения за конкретный день
- **Кумулятивные метрики** — накопленные значения за текущий месяц (помечены `_cum` или "кумулятивно")
- **28-дневные метрики** — значения за последние 28 дней

### Источники данных
Таблица формируется из следующих источников:
- `units_on_sk_platform_rep_mobile_total` — единицы на платформе
- `subscriptions_report_comerce_rep_mobile_total` — подписки и коммерция
- `total_active_users_per_day_rep_mobile_total` — активные пользователи за день
- `total_active_users_rep_mobile_total` — общие активные пользователи
- `mobile_report_rep_mobile_full` — дневные платежи
- `mobile_report_cum_rep_mobile_full` — кумулятивные платежи
- `total_activated_account_rep_mobile_full` — активированные аккаунты
- `new_users_pd_rep_mobile_total` — новые пользователи
- `maf_rep_mobile_total` — метрики MAF

### Ключевые особенности
- **Обновление:** ежедневно в 06:05 через материализованное представление
- **Размерность:** партнер × город × дата
- **Сортировка:** по дате отчета (DESC)
- **Связи:** с таблицами партнеров и установочных точек

## Частые срезы/фильтры
- По дате (`report_date`) — чаще всего **последняя доступная дата**
- По партнеру (`partner_uuid`) — анализ по конкретному партнеру
- По городу (`city`) — региональная аналитика
- По платформе — Android vs iOS метрики
- По типу платежа — App Store vs YooKassa
- По тарифам — анализ по ценовым сегментам (1₽, 35₽, 69₽, 85₽, 249₽, 499₽, 2390₽)

### Определение «последней даты»
Последняя дата — это максимальная `report_date`, для которой выполняется условие:  
`sum(paying_users) > 0` или `sum(total_active_users) > 0`

### Конец прошлого месяца (правило выборки даты)
Для запросов вида «на конец прошлого месяца» используй максимальную `report_date`, строго меньшую начала текущего месяца (`toStartOfMonth(today())`). Рекомендуемый шаблон выборки даты:
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM mobile_report_total
  WHERE report_date < toStartOfMonth(today())
    AND total_active_users > 0
)
```

Пример: «Количество активных пользователей на конец прошлого месяца» — суммируем по всем партнёрам и городам на выбранную дату:
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM mobile_report_total
  WHERE report_date < toStartOfMonth(today())
    AND total_active_users > 0
)
SELECT
  report_date AS `Дата`,
  sum(total_active_users) AS `Количество активных пользователей`
FROM mobile_report_total
INNER JOIN last_date USING (report_date);
```

## Примеры запросов (ClickHouse)

1) **Топ-10 партнеров по доходу за последний месяц**
```sql
WITH last_month AS (
  SELECT 
    toStartOfMonth(max(report_date)) AS start_date,
    max(report_date) AS end_date
  FROM mobile_report_total
  WHERE paying_users > 0
)
SELECT
  partner_uuid,
  sum(Android_PL + IOS_PL) AS total_revenue,
  sum(paying_users) AS total_subscriptions,
  sum(total_active_users) AS total_active_users
FROM mobile_report_total
CROSS JOIN last_month
WHERE report_date >= start_date AND report_date <= end_date
GROUP BY partner_uuid
ORDER BY total_revenue DESC
LIMIT 10;
```

2) **Динамика подписок по платформам за последние 30 дней**
```sql
SELECT
  report_date,
  sum(android_sub + android_sub_from_cart) AS android_subscriptions,
  sum(ios_sub + ios_sub_from_cart) AS ios_subscriptions,
  sum(Android_PL) AS android_revenue,
  sum(IOS_PL) AS ios_revenue
FROM mobile_report_total
WHERE report_date >= today() - 30
  AND paying_users > 0
GROUP BY report_date
ORDER BY report_date;
```

3) **Анализ возвратов по тарифам за месяц**
```sql
WITH current_month AS (
  SELECT toStartOfMonth(today()) AS start_date
)
SELECT
  'App Store' AS platform,
  sum(appstore_count_1_refunded) AS refunds_1_rub,
  sum(appstore_count_69_refunded) AS refunds_69_rub,
  sum(appstore_count_499_refunded) AS refunds_499_rub,
  sum(appstore_count_2390_refunded) AS refunds_2390_rub,
  sum(refunded_amount_appstore) AS total_refunded_amount
FROM mobile_report_total
CROSS JOIN current_month
WHERE report_date >= start_date
GROUP BY platform

UNION ALL

SELECT
  'YooKassa' AS platform,
  sum(yookassa_count_1_refunded) AS refunds_1_rub,
  sum(yookassa_count_69_refunded) AS refunds_69_rub,
  sum(yookassa_count_499_refunded) AS refunds_499_rub,
  sum(yookassa_count_2390_refunded) AS refunds_2390_rub,
  sum(refunded_amount_yookassa) AS total_refunded_amount
FROM mobile_report_total
CROSS JOIN current_month
WHERE report_date >= start_date
GROUP BY platform;
```

4) **Топ-10 городов по активным пользователям**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM mobile_report_total
  WHERE total_active_users > 0
)
SELECT
  city,
  sum(total_active_users) AS active_users,
  sum(total_active_users_monetization) AS monetized_users,
  sum(paying_users) AS paying_users,
  sum(units_on_platform) AS total_units
FROM mobile_report_total
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY active_users DESC
LIMIT 10;
```

5) **Конверсия в подписки по партнерам**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM mobile_report_total
  WHERE total_active_users > 0
)
SELECT
  partner_uuid,
  sum(total_active_users) AS total_users,
  sum(paying_users) AS paying_users,
  sum(paying_users) / nullIf(sum(total_active_users), 0) AS conversion_rate,
  sum(Android_PL + IOS_PL) AS revenue
FROM mobile_report_total
INNER JOIN last_date USING (report_date)
GROUP BY partner_uuid
HAVING total_users > 0
ORDER BY conversion_rate DESC
LIMIT 20;
```

6) **Анализ MAF (Monthly Active Flats) по городам**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM mobile_report_total
  WHERE MAF > 0
)
SELECT
  city,
  sum(MAF) AS total_maf,
  sum(stricted_MAF) AS stricted_maf,
  sum(freemonetization_MAF) AS free_maf,
  sum(freemonetization_MAF) / nullIf(sum(MAF), 0) AS free_maf_ratio
FROM mobile_report_total
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY total_maf DESC
LIMIT 15;
```

## Ключи и соединения
- **Основные ключи:** `partner_uuid`, `city`, `report_date`
- **Возможные связи:** 
  - К справочнику партнеров — по `partner_uuid`
  - К установочным точкам — через `installation_point_st_partner_ch`
  - К адресным данным — через `entries_installation_points_dir_partner`

## Ограничения и примечания
- **Обновление данных:** ежедневно в 06:05, возможны задержки до 1-2 часов
- **Дневные vs кумулятивные:** четко различайте дневные метрики от кумулятивных (помечены `_cum`)
- **Нулевые значения:** многие метрики могут быть NULL или 0, используйте `nullIf()` в расчетах
- **Платформы:** Android и iOS метрики разделены, учитывайте при анализе
- **Тарифы:** цены указаны в рублях, учитывайте инфляцию при долгосрочном анализе
- **BLE метрики:** связаны с Bluetooth Low Energy функциональностью
- **MAF метрики:** показывают активность на уровне квартир, а не пользователей
