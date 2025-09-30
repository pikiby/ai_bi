---
id: t_ai_global_report
title: Глобальная таблица AI по мобильным подпискам и платежам
short_description: "⚠️ КРИТИЧЕСКИ ВАЖНО: Это ТАБЛИЦА ClickHouse, НЕ дашборд! Ежедневная глобальная таблица, объединяющая все метрики мобильного бизнеса: подписки, платежи, активные пользователи, единицы на платформе, активированные аккаунты, новые пользователи и MAF с разбивкой по партнерам и городам."
type: table
tags:
  - "#KnowledgeBase"
  - "#AI_Report"
---

# «Глобальная таблица AI по мобильным подпискам и платежам» — t_ai_global_report

## Названия таблицы
**Короткое имя (человекочитаемое):** Глобальная таблица AI по мобильным подпискам и платежам  
**Тех. имя:** `t_ai_global_report`

### ⚠️ **КРИТИЧЕСКИ ВАЖНО - Это ТАБЛИЦА, НЕ ДАШБОРД**
**t_ai_global_report - это ТАБЛИЦА ClickHouse для SQL-запросов, НЕ дашборд!** К этой таблице можно и нужно писать SQL-запросы для получения данных.

## Назначение
**t_ai_global_report - это ТАБЛИЦА ClickHouse, НЕ дашборд!** Таблица `t_ai_global_report` является центральной агрегированной таблицей для AI-анализа мобильного бизнеса. Она объединяет ключевые метрики по подпискам, платежам, активным пользователям, единицам на платформе, активированным аккаунтам, новым пользователям и MAF (Monthly Active Flats) с детализацией по партнерам и городам. Таблица предназначена для предоставления комплексного ежедневного среза данных, необходимого для принятия стратегических решений и глубокой аналитики.

### ⚠️ **КРИТИЧЕСКИ ВАЖНО - Обогащенные данные**
**Таблица УЖЕ содержит обогащенные данные!** Поля `company_name`, `country`, `region` уже доступны напрямую в таблице и НЕ требуют дополнительных JOIN к справочникам. Используйте эти поля напрямую в SELECT без обогащения через `companies_dir_partner_ch` или другие справочники.

### ⚠️ **КРИТИЧЕСКИ ВАЖНО - Работа с датами**
**ПРАВИЛО ПО УМОЛЧАНИЮ:** Если пользователь НЕ указал дату в запросе, ВСЕГДА используй последнюю доступную дату из таблицы: `WHERE report_date = (SELECT max(report_date) FROM t_ai_global_report WHERE total_active_users > 0)`

**Основные метрики:**

### 📱 **Подписки и коммерция:**
- **Android подписки:** новые, продленные, с карты, через YooKassa
- **iOS подписки:** через App Store, с карты, переходы между платформами
- **Коммерческие метрики:** активированные подписки, PRO подписки, стандартные подписки
- **Ошибки и отмены:** неудачные продления, отмены подписок по платформам
- **28-дневные метрики:** новые пользователи, продления, отмены за последние 28 дней

### 💰 **Платежи и доходы:**
- **Дневные платежи:** прибыль от Android/iOS, покупки по тарифам (1₽, 35₽, 69₽, 85₽, 249₽, 499₽, 2390₽)
- **Кумулятивные платежи:** накопленные значения за текущий месяц
- **Возвраты:** суммы и количество возвратов по платформам и тарифам
- **Платформенная аналитика:** App Store vs YooKassa, Android vs iOS

### 👥 **Активные пользователи:**
- **Общие метрики:** общее количество активных пользователей за день
- **Монетизация:** пользователи с монетизацией, с подпиской и монетизацией
- **BLE функциональность:** пользователи с доступным Bluetooth Low Energy
- **Новые пользователи:** активированные в текущем месяце

### 🏢 **Единицы на платформе:**
- **Свободная монетизация:** единицы со свободной монетизацией, PRO подписки
- **Ограниченная монетизация:** единицы с ограниченной монетизацией
- **Общие метрики:** общее количество единиц на платформе

### 🎯 **Активированные аккаунты:**
- **Общие метрики:** общее количество активированных аккаунтов
- **Специализированные:** с монетизацией, с BLE, с BLE и монетизацией
- **Новые аккаунты:** созданные и активированные за день/месяц

### 🏠 **MAF (Monthly Active Flats):**
- **Уникальные квартиры:** с активными пользователями за день
- **Монетизация квартир:** с ограниченной/свободной монетизацией

<!-- ### 🔑 **Лицензионные ключи (временно отключены):**
- **Активированные ключи:** общее количество активированных ключей
- **Подписчики с ключами:** вернувшиеся и новые подписчики с ключами
- **Отмены:** отменившие подписку (вернувшиеся и новые)

### 📍 **Установочные точки (временно отключены):**
- **Количество точек:** общее количество точек установки
- **Онлайн статус:** точки с домофоном онлайн, количество домофонов онлайн
- **Домофоны:** общее количество домофонов
-->

### 🏢 **Реквизиты партнеров:**
- **ИНН:** налоговый номер партнера
- **Личный кабинет:** идентификатор личного кабинета партнера

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
        eipdp.country AS country,
        eipdp.region AS region,
        cdp.company_name AS company_name,
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
**ВНИМАНИЕ: Поля `company_name`, `country`, `region` уже обогащены и доступны напрямую - НЕ требуют JOIN!**

| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `report_date`                 | Date    | Дата данных                               |
| `partner_uuid`                | String  | UUID партнера                             |
| `city_uuid`                   | String  | UUID города                               |
| `city`                        | String  | Город                                     |
| `country`                     | String  | Страна                                    |
| `region`                      | String  | Регион                                    |
| `company_name`                | String  | Название компании                          |
| `partner_lk`                  | String  | Личный кабинет партнёра                   |

### Метрики подписок
**ВНИМАНИЕ: Все поля содержат количество подписок, а не суммы оплат!**

| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `android_sub`                 | UInt64  | Количество подписок Android (старые данные)         |
| `android_sub_extended_new`    | UInt64  | Количество продленных подписок Android (старые данные) |
| `android_sub_extended_new_cart` | UInt64 | Количество продленных подписок Android с карты      |
| `android_sub_first_new`      | UInt64  | Количество новых подписок Android (старые данные)    |
| `android_sub_first_new_cart` | UInt64  | Количество новых подписок Android с карты            |
| `android_sub_from_cart`       | UInt64  | Количество подписок Android с карты                  |
| `ios_sub`                    | UInt64  | Количество подписок iOS через App Store              |
| `ios_sub_extended_new`       | UInt64  | Количество продленных подписок iOS через App Store   |
| `ios_sub_extended_new_cart`  | UInt64  | Количество продленных подписок iOS с карты           |
| `ios_sub_first_new`          | UInt64  | Количество новых подписок iOS через App Store        |
| `ios_sub_first_new_cart`     | UInt64  | Количество новых подписок iOS с карты                |
| `ios_sub_first_new_cart_transition` | UInt64 | Количество новых подписок iOS с карты (переход из App Store) |
| `ios_sub_from_cart`          | UInt64  | Количество подписок iOS с карты                      |
| `paying_users`               | UInt64  | Количество всех активированных подписок               |
| `paying_users_day`           | UInt64  | Количество подписок, активированных в день данных    |
| `paying_users_partner_pro`   | UInt64  | Количество PRO подписок партнеров                    |
| `paying_users_standart`      | UInt64  | Количество стандартных подписок                      |
| `paying_users_standart_appstore` | UInt64 | Количество подписок iOS через App Store             |
| `paying_users_standart_ios_from_cart` | UInt64 | Количество подписок iOS с карты (переход с App Store) |
| `paying_users_standart_yakassa` | UInt64 | Количество подписок Android через YooKassa (старые данные) |

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

<!-- ### Метрики лицензионных ключей (временно отключены)
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `activated_keys`              | UInt32  | Активированные ключи                      |
| `activated_sub_w_key`         | UInt32  | Вернувшиеся подписчики с ключом           |
| `created_first_sub_w_key`     | UInt32  | Новые подписчики с ключом                 |
| `d_activated_sub_w_key`       | UInt32  | Отменившие подписку (вернувшиеся)         |
| `d_created_first_sub_w_key`  | UInt32  | Отменившие подписку (новые)               |

### Метрики установочных точек (временно отключены)
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `countd_inst_by_c_and_p`     | UInt64  | Количество точек установки                |
| `countd_inter_par_by_c_and_p` | UInt64  | Количество точек установки с домофоном онлайн |
| `countd_inst_par_onl_by_c_and_p` | UInt64 | Количество домофонов                      |
| `countd_inter_par_onl_by_c_and_p` | UInt64 | Количество домофонов онлайн               |
-->

### Реквизиты партнеров
| Поле                          | Тип     | Алиас (человекочитаемое)                  |
| ----------------------------- | ------- | ----------------------------------------- |
| `tin`                         | String  | ИНН                                       |

## Примечания к полям

### Типы метрик
- **Дневные метрики** — значения за конкретный день
- **Кумулятивные метрики** — накопленные значения за текущий месяц (помечены `_cum` или "кумулятивно")
- **28-дневные метрики** — значения за последние 28 дней


### Ключевые особенности
- **Обновление:** ежедневно в 06:05 через материализованное представление с фильтрацией `WHERE report_date < today() - 2`
- **Размерность:** партнер × город × дата (декартово произведение всех комбинаций)
- **Сортировка:** по дате данных (DESC) для быстрого доступа к последним данным
- **Связи:** с таблицами партнеров, установочных точек и справочниками городов
- **AI-оптимизация:** структура оптимизирована для AI-анализа и машинного обучения
- **Исторические данные:** поддержка загрузки данных за произвольные периоды через INSERT запросы
- **Стабильность данных:** исключение неполных данных за последние 2 дня
- **Комплексность:** объединение 10+ источников данных в единую структуру (лицензионные ключи и установочные точки временно отключены)

## Частые срезы/фильтры

### 📅 **Временные срезы:**
- **По дате (`report_date`):** чаще всего последняя доступная дата
- **По периоду:** диапазоны дат для трендового анализа
- **По месяцам:** кумулятивные метрики за текущий месяц
- **28-дневные окна:** для анализа недавней активности

### 🏢 **Организационные срезы:**
- **По партнеру (`partner_uuid`):** анализ по конкретному партнеру
- **По городу (`city`, `city_uuid`):** региональная аналитика
- **По личному кабинету (`partner_lk`):** анализ по ЛК партнера
- **По ИНН (`tin`):** анализ по налоговому номеру

### 📱 **Платформенные срезы:**
- **Android vs iOS:** сравнение метрик по платформам
- **App Store vs YooKassa:** анализ платежных систем
- **Карты vs платформенные платежи:** анализ способов оплаты

### 💰 **Финансовые срезы:**
- **По тарифам:** анализ по ценовым сегментам (1₽, 35₽, 69₽, 85₽, 249₽, 499₽, 2390₽)
- **По доходам:** дневные vs кумулятивные метрики
- **По возвратам:** анализ возвратов по платформам и тарифам

### 👥 **Пользовательские срезы:**
- **Активные пользователи:** общие, с монетизацией, с BLE
- **Новые пользователи:** созданные vs активированные
- **Подписчики:** с ключами, без ключей, по типам подписок

### 🏠 **Географические срезы:**
- **MAF метрики:** активность на уровне квартир
- **Установочные точки:** анализ по точкам установки
- **Домофоны:** онлайн статус, количество домофонов

### Определение «последней даты»
Последняя дата — это максимальная `report_date`, для которой выполняется условие:  
`sum(paying_users) > 0` или `sum(total_active_users) > 0`

### ⚠️ **КРИТИЧЕСКИ ВАЖНО - Работа с датами**

**ПРАВИЛО ПО УМОЛЧАНИЮ:** Если пользователь НЕ указал дату в запросе, ВСЕГДА используй последнюю доступную дату из таблицы.

**Логика выбора даты:**
- ✅ **Если пользователь указал конкретную дату** — используй её: `WHERE report_date = '2025-08-31'`
- ✅ **Если пользователь указал период** — используй диапазон: `WHERE report_date BETWEEN '2025-08-01' AND '2025-08-31'`
- ✅ **Если пользователь НЕ указал дату** — используй последнюю доступную: `WHERE report_date = (SELECT max(report_date) FROM t_ai_global_report WHERE total_active_users > 0)`

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

6) **Анализ установочных точек и домофонов по городам и партнерам**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  partner_lk,
  sum(countd_inst_by_c_and_p) AS total_installation_points,
  sum(countd_inter_par_by_c_and_p) AS points_with_online_intercoms,
  sum(countd_inst_par_onl_by_c_and_p) AS total_intercoms,
  sum(countd_inter_par_onl_by_c_and_p) AS online_intercoms,
  sum(countd_inter_par_onl_by_c_and_p) / nullIf(sum(countd_inst_par_onl_by_c_and_p), 0) AS intercom_online_rate
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city, partner_lk
ORDER BY total_installation_points DESC
LIMIT 20;
```

7) **Анализ лицензионных ключей и подписок**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  partner_lk,
  sum(activated_keys) AS total_activated_keys,
  sum(activated_sub_w_key) AS returning_subscribers_with_key,
  sum(created_first_sub_w_key) AS new_subscribers_with_key,
  sum(d_activated_sub_w_key) AS cancelled_returning_subscribers,
  sum(d_created_first_sub_w_key) AS cancelled_new_subscribers,
  sum(activated_sub_w_key + created_first_sub_w_key) / nullIf(sum(activated_keys), 0) AS key_to_subscription_conversion
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city, partner_lk
HAVING total_activated_keys > 0
ORDER BY key_to_subscription_conversion DESC
LIMIT 15;
```

8) **Комплексный анализ эффективности партнеров**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  partner_lk,
  city,
  sum(total_active_users) AS active_users,
  sum(paying_users) AS paying_users,
  sum(Android_PL + IOS_PL) AS revenue,
  sum(countd_inst_by_c_and_p) AS installation_points,
  sum(countd_inter_par_onl_by_c_and_p) AS online_intercoms,
  sum(activated_keys) AS activated_keys,
  sum(activated_sub_w_key + created_first_sub_w_key) AS subscribers_with_keys,
  sum(paying_users) / nullIf(sum(total_active_users), 0) AS user_conversion,
  sum(Android_PL + IOS_PL) / nullIf(sum(paying_users), 0) AS revenue_per_paying_user,
  sum(countd_inter_par_onl_by_c_and_p) / nullIf(sum(countd_inst_par_onl_by_c_and_p), 0) AS intercom_online_rate
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY partner_lk, city
HAVING active_users > 0
ORDER BY revenue DESC
LIMIT 25;
```

9) **Детальный анализ платежей по тарифам и платформам**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  city,
  partner_lk,
  -- iOS App Store метрики
  sum(appstore_count_1) AS appstore_1_rub,
  sum(appstore_count_35) AS appstore_35_rub,
  sum(appstore_count_69) AS appstore_69_rub,
  sum(appstore_count_85) AS appstore_85_rub,
  sum(appstore_count_249) AS appstore_249_rub,
  sum(appstore_count_499) AS appstore_499_rub,
  sum(appstore_count_2390) AS appstore_2390_rub,
  sum(IOS_PL) AS ios_total_revenue,
  -- Android YooKassa метрики
  sum(yookassa_count_1) AS yookassa_1_rub,
  sum(yookassa_count_35) AS yookassa_35_rub,
  sum(yookassa_count_69) AS yookassa_69_rub,
  sum(yookassa_count_85) AS yookassa_85_rub,
  sum(yookassa_count_249) AS yookassa_249_rub,
  sum(yookassa_count_499) AS yookassa_499_rub,
  sum(yookassa_count_2390) AS yookassa_2390_rub,
  sum(Android_PL) AS android_total_revenue,
  -- Возвраты
  sum(refunded_amount_appstore) AS appstore_refunds,
  sum(refunded_amount_yookassa) AS yookassa_refunds
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
GROUP BY city, partner_lk
ORDER BY (ios_total_revenue + android_total_revenue) DESC
LIMIT 20;
```

10) **Анализ популярности тарифов по платформам**
```sql
WITH last_date AS (
  SELECT max(report_date) AS report_date
  FROM t_ai_global_report
  WHERE total_active_users > 0
)
SELECT
  'iOS App Store' AS platform,
  sum(appstore_count_1) AS tariff_1_rub,
  sum(appstore_count_35) AS tariff_35_rub,
  sum(appstore_count_69) AS tariff_69_rub,
  sum(appstore_count_85) AS tariff_85_rub,
  sum(appstore_count_249) AS tariff_249_rub,
  sum(appstore_count_499) AS tariff_499_rub,
  sum(appstore_count_2390) AS tariff_2390_rub,
  sum(IOS_PL) AS total_revenue
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)

UNION ALL

SELECT
  'Android YooKassa' AS platform,
  sum(yookassa_count_1) AS tariff_1_rub,
  sum(yookassa_count_35) AS tariff_35_rub,
  sum(yookassa_count_69) AS tariff_69_rub,
  sum(yookassa_count_85) AS tariff_85_rub,
  sum(yookassa_count_249) AS tariff_249_rub,
  sum(yookassa_count_499) AS tariff_499_rub,
  sum(yookassa_count_2390) AS tariff_2390_rub,
  sum(Android_PL) AS total_revenue
FROM t_ai_global_report
INNER JOIN last_date USING (report_date)
ORDER BY total_revenue DESC;
```

11) **Месячный анализ доходов с детализацией по тарифам**
```sql
WITH month_bounds AS (
  SELECT toStartOfMonth(today()) AS m_start, 
         addMonths(toStartOfMonth(today()), 1) AS m_end
)
SELECT
  partner_lk,
  city,
  sum(IOS_PL) AS ios_monthly_revenue,
  sum(Android_PL) AS android_monthly_revenue,
  sum(IOS_PL + Android_PL) AS total_monthly_revenue,
  -- Детализация по тарифам iOS
  sum(appstore_count_1 * 1) AS ios_1_rub_total,
  sum(appstore_count_35 * 35) AS ios_35_rub_total,
  sum(appstore_count_69 * 69) AS ios_69_rub_total,
  sum(appstore_count_85 * 85) AS ios_85_rub_total,
  sum(appstore_count_249 * 249) AS ios_249_rub_total,
  sum(appstore_count_499 * 499) AS ios_499_rub_total,
  sum(appstore_count_2390 * 2390) AS ios_2390_rub_total,
  -- Детализация по тарифам Android
  sum(yookassa_count_1 * 1) AS android_1_rub_total,
  sum(yookassa_count_35 * 35) AS android_35_rub_total,
  sum(yookassa_count_69 * 69) AS android_69_rub_total,
  sum(yookassa_count_85 * 85) AS android_85_rub_total,
  sum(yookassa_count_249 * 249) AS android_249_rub_total,
  sum(yookassa_count_499 * 499) AS android_499_rub_total,
  sum(yookassa_count_2390 * 2390) AS android_2390_rub_total,
  -- Возвраты
  sum(refunded_amount_appstore) AS total_appstore_refunds,
  sum(refunded_amount_yookassa) AS total_yookassa_refunds
FROM t_ai_global_report
CROSS JOIN month_bounds
WHERE report_date >= m_start AND report_date < m_end
GROUP BY partner_lk, city
ORDER BY total_monthly_revenue DESC
LIMIT 15;
```

## Ключи и соединения
- **Основные ключи:** `partner_uuid`, `city_uuid`, `report_date`
- **Возможные связи:** 
  - К справочнику партнеров — по `partner_uuid`
  - К установочным точкам — через `installation_point_st_partner_ch`
  - К адресным данным — через `entries_installation_points_dir_partner_ch`

## Ограничения и примечания

### ⏰ **Временные ограничения:**
- **Обновление данных:** ежедневно в 06:05, возможны задержки до 1-2 часов
- **Фильтрация:** исключение данных за последние 2 дня для стабильности
- **Исторические данные:** поддержка загрузки данных за произвольные периоды

### 📊 **Типы метрик:**
- **Дневные vs кумулятивные:** четко различайте дневные метрики от кумулятивных (помечены `_cum`)
- **28-дневные окна:** метрики за последние 28 дней для анализа недавней активности
- **Месячные агрегаты:** кумулятивные значения накопленные за текущий месяц

### 🔢 **Обработка данных:**
- **Нулевые значения:** многие метрики могут быть NULL или 0, используйте `nullIf()` в расчетах
- **Декартово произведение:** таблица содержит все комбинации партнер × город × дата
- **LEFT JOIN:** все источники данных подключены через LEFT JOIN для сохранения всех записей

### 📱 **Платформенные особенности:**
- **Android vs iOS:** метрики разделены по платформам, учитывайте при анализе
- **App Store vs YooKassa:** разные платежные системы с разными тарифами
- **Карты vs платформенные платежи:** различные способы оплаты подписок

### 💰 **Финансовые особенности:**
- **Тарифы:** цены указаны в рублях, учитывайте инфляцию при долгосрочном анализе
- **Возвраты:** отдельные метрики для возвратов по платформам и тарифам
- **Кумулятивные доходы:** накопленные значения за месяц для трендового анализа

### 🔧 **Технические особенности:**
- **BLE метрики:** связаны с Bluetooth Low Energy функциональностью
- **MAF метрики:** показывают активность на уровне квартир, а не пользователей
- **Лицензионные ключи:** метрики по активации и использованию ключей
- **Установочные точки:** анализ по точкам установки и домофонам

### 🤖 **AI-оптимизация:**
- **Структура:** таблица специально структурирована для AI-анализа и машинного обучения
- **Комплексность:** объединение 10+ источников данных в единую структуру (лицензионные ключи и установочные точки временно отключены)
- **Масштабируемость:** поддержка больших объемов данных с оптимизированными запросами

### Запрещено (ClickHouse-совместимость)
- Писать агрегаты в WHERE верхнего запроса. Получите ILLEGAL_AGGREGATION.
- Давать алиас агрегату с тем же именем, что и поле таблицы, и затем использовать его в WHERE/SELECT (например, `AS report_date`). Используйте нейтральные алиасы (`last_date`, `end_date`).
- Исключать список значений через подзапрос с `arrayJoin(...)` прямо в WHERE. Используйте `NOT IN ('...')` или `NOT has([...], col)`.
