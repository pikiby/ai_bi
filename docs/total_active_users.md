# Таблица db1.total_active_users_rep_mobile_total

Эта таблица хранит ежедневную статистику активных пользователей мобильного приложения.

## Назначение
- Аналитика общего количества активных пользователей.
- Разделение на новые и существующие аккаунты.
- Учёт монетизации и подписок.
- Учёт доступности BLE-функционала.

## Структура таблицы (DDL)

```sql
CREATE TABLE db1.total_active_users_rep_mobile_total
(
    `report_date` Date,  
    `partner_uuid` String,  -- идентификатор партнёра
    `city` String,  -- название города
    `total_active_users` UInt32,  -- всего активных пользователей
    `new_active_users` UInt32,  -- новых активных пользователей
    `total_active_users_monetization` UInt32,  -- пользователи с монетизацией
    `total_active_user_subscribed_monetization` UInt32,  -- подписчики с монетизацией
    `total_active_users_ble_available` UInt32,  -- пользователи с BLE-доступом
    `total_active_users_ble_available_monetization` UInt32,  -- пользователи BLE с монетизацией
    `total_active_users_ble_available_subscribed_monetization` UInt32  -- подписчики BLE с монетизацией
)
ENGINE = MergeTree()
ORDER BY partner_uuid

## Алиасы столбцов

`report_date` - дата отчёта 
`partner_uuid`- идентификатор партнёра
`city` - название города
`total_active_users` - всего активных пользователей
`new_active_users` - новых активных пользователей
`total_active_users_monetization` - пользователи с монетизацией
`total_active_user_subscribed_monetization` - подписчики с монетизацией
`total_active_users_ble_available` - пользователи с BLE-доступом
`total_active_users_ble_available_monetization` - пользователи BLE с монетизацией
`total_active_users_ble_available_subscribed_monetization` - подписчики BLE с монетизацией
