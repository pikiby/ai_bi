# «Таблица активных пользователей мобильного приложения» — db1.total_active_users_rep_mobile_total

## Названия таблицы

**Короткое имя (человекочитаемое):** Таблица активных пользователей мобильного приложения  
**Тех. имя:** `db1.total_active_users_rep_mobile_total`

## Назначение
Ежедневная статистика активных пользователей мобильного приложения: общее число, новые, монетизация, подписки, доступность BLE.

## Хранилище и движок
- БД: `db1`
- Таблица: `total_active_users_rep_mobile_total`
- Движок: `MergeTree`
- Сортировка: `ORDER BY partner_uuid`

## DDL
```sql
CREATE TABLE db1.total_active_users_rep_mobile_total
(
    `report_date` Date,
    `partner_uuid` String,
    `city` String,
    `total_active_users` UInt32,
    `new_active_users` UInt32,
    `total_active_users_monetization` UInt32,
    `total_active_user_subscribed_monetization` UInt32,
    `total_active_users_ble_available` UInt32,
    `total_active_users_ble_available_monetization` UInt32,
    `total_active_users_ble_available_subscribed_monetization` UInt32
)
ENGINE = MergeTree()
ORDER BY partner_uuid
```

## Поля и алиасы
| Поле                                                | Тип     | Алиас (человекочитаемое)                           |
|---                                                  |---      |---                                                 |
| `report_date`                                       | `Date`  | `дата отчёта`                                      |
| `partner_uuid`                                      | `String`| `идентификатор партнёра`                           |
| `city`                                              | `String`| `название города`                                  |
| `total_active_users`                                | `UInt32`| `всего активных пользователей`                     |
| `new_active_users`                                  | `UInt32`| `новых активных пользователей за месяц`                     |
| `total_active_users_monetization`                   | `UInt32`| `пользователи с монетизацией`                      |
| `total_active_user_subscribed_monetization`         | `UInt32`| `подписчики с монетизацией`                        |
| `total_active_users_ble_available`                  | `UInt32`| `пользователи с BLE-доступом`                      |
| `total_active_users_ble_available_monetization`     | `UInt32`| `пользователи BLE с монетизацией`                  |
| `total_active_users_ble_available_subscribed_monetization` | `UInt32`| `подписчики BLE с монетизацией`             |

> Используйте алиасы в SELECT через `AS` и заключайте их в обратные кавычки, например: `total_active_users AS \`всего активных пользователей\``.

## Частые срезы/фильтры
- По дате: `report_date`
- По партнёру: `partner_uuid`
- По городу: `city`

## Примеры запросов (с алиасами)
1) Общее число активных пользователей по партнёру за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.total_active_users_rep_mobile_total
)
SELECT
  t.partner_uuid AS `идентификатор партнёра`,
  t.total_active_users AS `всего активных пользователей`
FROM db1.total_active_users_rep_mobile_total AS t
INNER JOIN max_dt USING(report_date)
```

2) Разбивка по городам за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM db1.total_active_users_rep_mobile_total
)
SELECT
  t.city AS `название города`,
  t.total_active_users AS `всего активных пользователей`,
  t.new_active_users AS `новых активных пользователей`,
  t.total_active_users_monetization AS `пользователи с монетизацией`
FROM db1.total_active_users_rep_mobile_total AS t
INNER JOIN max_dt USING(report_date)
ORDER BY 2 DESC
```

## Ключи и соединения
- Сортировка `ORDER BY partner_uuid`. Рекомендуется использовать `partner_uuid` для группировок/фильтров.
- Внешние связи не указаны. При необходимости допишите рекомендуемые ключи JOIN.

## Ограничения и примечания
- В документе нет описаний NULL-политики/дефолтов, источников и SLA обновления.
- Партиционирование в DDL не указано.
mobile_report_rep_mobile_full
- Поля имеют накопительный характер **в пределах месяца**; для значения «за месяц» берите строку с максимальной датой в этом месяце (`max(report_date)` при фильтре по месяцу). Разности по датам считать **не нужно**.
