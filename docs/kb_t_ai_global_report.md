---
id: t_ai_global_report
title: Глобальный AI отчет по мобильным подпискам и платежам
short_description: "Ежедневный глобальный отчет, объединяющий все метрики мобильного бизнеса: подписки, платежи, активные пользователи, единицы на платформе, активированные аккаунты, новые пользователи и MAF с разбивкой по партнерам и городам."
type: table
tags:
  - "#KnowledgeBase"
  - "#AI_Report"
---

# «Глобальный AI отчет по мобильным подпискам и платежам» — t_ai_global_report

## Названия таблицы
**Короткое имя (человекочитаемое):** Глобальный AI отчет по мобильным подпискам и платежам  
**Тех. имя:** `t_ai_global_report`

## Назначение
Таблица предоставляет ежедневный глобальный срез всех ключевых метрик мобильного бизнеса, объединяя данные из всех специализированных отчетов. Используется для комплексной аналитики, AI-анализа, мониторинга бизнес-показателей и принятия стратегических решений.

**Основные метрики:**
- Подписки (Android/iOS, новые/продленные, через App Store/карты)
- Платежи (доходы, покупки, возвраты по платформам и тарифам)
- Активные пользователи (общие, с монетизацией, с BLE)
- Единицы на платформе (свободная/ограниченная монетизация)
- Активированные аккаунты (общие, с монетизацией, с BLE)
- Новые пользователи (созданные/активированные)
- MAF (Monthly Active Flats)
- Лицензионные ключи (активированные, подписчики с ключами)
- Установочные точки (количество точек, домофонов, онлайн статус)
- Реквизиты партнеров (ИНН)

## Хранилище и движок
- БД: `db1`  
- Таблица: `t_ai_global_report`  
- Движок: `MergeTree`  
- Сортировка: `ORDER BY report_date`
- Материализованное представление: `t_ai_global_report_mv` (обновляется ежедневно в 06:05)

## DDL
```sql
CREATE MATERIALIZED VIEW db1.t_ai_global_report_mv
    REFRESH EVERY 1 DAY OFFSET 6 HOUR 5 MINUTE TO db1.t_ai_global_report AS
    WITH t_dec AS(
        SELECT
            t1.report_date  AS report_date,
            t2.city_uuid  AS city_uuid,
            t2.partner_uuid  AS partner_uuid
        FROM
            (SELECT DISTINCT report_date FROM db1.installation_point_st_partner_ch) AS t1
        CROSS JOIN
            (SELECT DISTINCT city_uuid, inst.partner_uuid AS partner_uuid
            FROM db1.installation_point_st_partner_ch AS inst
            LEFT JOIN db1.entries_installation_points_dir_partner_ch AS eipdp 
                ON inst.installation_point_id = eipdp.installation_point_id) AS t2
        )
    SELECT
        t_dec.report_date AS report_date,
        t_dec.partner_uuid AS partner_uuid,
        t_dec.city_uuid AS city_uuid,
        eipdp.city AS city,
        cdp.partner_lk AS partner_lk,
        -- Все метрики из объединяемых таблиц
    FROM t_dec
    LEFT JOIN db1.units_on_sk_platform_rep_mobile_total AS uosprmt 
        ON uosprmt.report_date = t_dec.report_date 
        AND uosprmt.city_uuid = t_dec.city_uuid
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
| `city_uuid`                   | String  | UUID города                               |
| `city`                        | String  | Город                                     |
| `partner_lk`                  | String  | Личный кабинет партнёра                   |

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
| `renewed_subscriptions_last_28_days` | UInt64 | Продленные подписки за последние 28 дней |
| `stoped_subscript_last_28_days` | UInt64 | Отмены подписок за последние 28 дней      |

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
| `yookassa_count_69_refunded` | UInt64 | Возвраты за 69₽ через YooKassa за день   |
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
| `new_created_account`         | UInt64  | КУМУЛЯТИВНОЕ количество новых созданных аккаунтов за месяц |
| `new_activated_account`       | UInt64  | КУМУЛЯТИВНОЕ количество новых активированных аккаунтов за месяц |
| `new_created_account_day`     | UInt64  | Новые созданные аккаунты за день          |
| `new_activated_account_day`   | UInt64  | Новые активированные аккаунты за день     |

### Метрики MAF (Monthly Active Flats)
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `MAF`                         | UInt64  | Уникальные квартиры с активными пользователями за день |
| `stricted_MAF`               | UInt64  | Квартиры с ограниченной монетизацией      |
| `freemonetization_MAF`       | UInt64  | Квартиры со свободной монетизацией        |

### Метрики лицензионных ключей
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `activated_keys`              | UInt32  | Активированные ключи                      |
| `activated_sub_w_key`         | UInt32  | Вернувшиеся подписчики с ключом           |
| `created_first_sub_w_key`     | UInt32  | Новые подписчики с ключом                 |
| `d_activated_sub_w_key`       | UInt32  | Отменившие подписку (вернувшиеся)         |
| `d_created_first_sub_w_key`  | UInt32  | Отменившие подписку (новые)               |

### Метрики установочных точек
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `countd_inst_by_c_and_p`     | UInt64  | Количество точек установки                |
| `countd_inter_par_by_c_and_p` | UInt64  | Количество точек установки с домофоном онлайн |
| `countd_inst_par_onl_by_c_and_p` | UInt64 | Количество домофонов                      |
| `countd_inter_par_onl_by_c_and_p` | UInt64 | Количество домофонов онлайн               |

### Реквизиты партнеров
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `tin`                         | String  | ИНН                                       |

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
- `t_subscriptions_after_licensed_keys` — метрики лицензионных ключей
- `t_int_p_and_inst_by_cit_and_par` — метрики установочных точек
- `companies_dir_partner_ch` — реквизиты партнёров (в том числе `partner_lk`, `tin`)
- `entries_installation_points_dir_partner_ch` — справочник городов

### Ключевые особенности
- **Обновление:** ежедневно в 06:05 через материализованное представление
- **Размерность:** партнер × город × дата
- **Сортировка:** по дате отчета (DESC)
- **Связи:** с таблицами партнеров и установочных точек
- **AI-оптимизация:** структура оптимизирована для AI-анализа и машинного обучения

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

### Работа с датами
Для запросов с датами используй только явные даты, указанные пользователем:
- Если пользователь указал конкретную дату — используй её: `WHERE report_date = '2025-08-31'`
- Если пользователь указал период — используй диапазон: `WHERE report_date BETWEEN '2025-08-01' AND '2025-08-31'`
- Если пользователь НЕ указал дату — используй последнюю доступную: `WHERE report_date = (SELECT max(report_date) FROM t_ai_global_report WHERE total_active_users > 0)`

**Запрещено:** любые CTE с агрегатами (`WITH ... AS (SELECT max(...) ...)`) — они вызывают ILLEGAL_AGGREGATION в ClickHouse.

## Примеры запросов (ClickHouse)

1) **Комплексный анализ бизнес-метрик за последние 30 дней**
```sql
SELECT
  report_date,
  sum(total_active_users) AS active_users,
  sum(paying_users) AS paying_users,
  sum(Android_PL + IOS_PL) AS total_revenue,
  sum(units_on_platform) AS total_units,
  sum(MAF) AS total_maf
FROM t_ai_global_report
WHERE report_date >= today() - 30
  AND total_active_users > 0
GROUP BY report_date
ORDER BY report_date;
```

2) **AI-анализ конверсии по партнерам**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  partner_uuid,
  partner_lk,
  sum(total_active_users) AS total_users,
  sum(paying_users) AS paying_users,
  sum(paying_users) / nullIf(sum(total_active_users), 0) AS conversion_rate,
  sum(Android_PL + IOS_PL) AS revenue,
  sum(units_on_platform) AS units,
  sum(MAF) AS maf
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY partner_uuid, partner_lk
HAVING total_users > 0
ORDER BY conversion_rate DESC
LIMIT 20;
```

3) **Региональный анализ с AI-метриками**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  sum(total_active_users) AS active_users,
  sum(paying_users) AS paying_users,
  sum(Android_PL + IOS_PL) AS revenue,
  sum(units_on_platform) AS units,
  sum(MAF) AS maf,
  sum(new_created_account_day) AS new_accounts,
  sum(new_activated_account_day) AS new_activated
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city
ORDER BY active_users DESC
LIMIT 15;
```

4) **AI-анализ платформенных предпочтений**
```sql
SELECT
  'Android' AS platform,
  sum(android_sub + android_sub_from_cart) AS subscriptions,
  sum(Android_PL) AS revenue,
  sum(yookassa_count_1 + yookassa_count_35 + yookassa_count_69 + yookassa_count_85 + yookassa_count_249 + yookassa_count_499 + yookassa_count_2390) AS purchases
FROM t_ai_global_report
WHERE report_date >= toStartOfMonth(today())
  AND paying_users > 0

UNION ALL

SELECT
  'iOS' AS platform,
  sum(ios_sub + ios_sub_from_cart) AS subscriptions,
  sum(IOS_PL) AS revenue,
  sum(appstore_count_1 + appstore_count_69 + appstore_count_85 + appstore_count_499 + appstore_count_2390) AS purchases
FROM t_ai_global_report
WHERE report_date >= toStartOfMonth(today())
  AND paying_users > 0;
```

5) **AI-анализ роста пользователей**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  sum(total_active_users) AS current_active_users,
  sum(new_active_users) AS new_users_this_month,
  sum(new_created_account_day) AS new_accounts_today,
  sum(new_activated_account_day) AS new_activated_today,
  sum(total_activated_account) AS total_activated_accounts,
  sum(total_activated_account_monetization) AS monetized_accounts
FROM t_ai_global_report
INNER JOIN last_date USING (report_date);
```

## Ключи и соединения
- **Основные ключи:** `partner_uuid`, `city_uuid`, `report_date`
- **Возможные связи:** 
  - К справочнику партнеров — по `partner_uuid`
  - К установочным точкам — через `installation_point_st_partner_ch`
  - К адресным данным — через `entries_installation_points_dir_partner_ch`

## Ограничения и примечания
- **Обновление данных:** ежедневно в 06:05, возможны задержки до 1-2 часов
- **Дневные vs кумулятивные:** четко различайте дневные метрики от кумулятивных (помечены `_cum`)
- **Нулевые значения:** многие метрики могут быть NULL или 0, используйте `nullIf()` в расчетах
- **Платформы:** Android и iOS метрики разделены, учитывайте при анализе
- **Тарифы:** цены указаны в рублях, учитывайте инфляцию при долгосрочном анализе
- **BLE метрики:** связаны с Bluetooth Low Energy функциональностью
- **MAF метрики:** показывают активность на уровне квартир, а не пользователей
- **AI-оптимизация:** таблица специально структурирована для AI-анализа и машинного обучения

### Запрещено (ClickHouse-совместимость)
- Писать агрегаты в WHERE верхнего запроса. Получите ILLEGAL_AGGREGATION.
- Давать алиас агрегату с тем же именем, что и поле таблицы, и затем использовать его в WHERE/SELECT (например, `AS report_date`). Используйте нейтральные алиасы (`last_date`, `end_date`).
- Исключать список значений через подзапрос с `arrayJoin(...)` прямо в WHERE. Используйте `NOT IN ('...')` или `NOT has([...], col)`.
