---
id: partners_monetization_status
title: Статус монетизации партнеров в день
db: default
short_description: "Ежедневная статистика блокировки монетизации у партнеров по точкам установки; включает процент заблокированных точек и детализацию по городам."
synonyms:
- статус монетизации партнеров
- блокировка монетизации
- точки установки партнеров
- процент заблокированных точек
- монетизация по партнерам
type: table
tags: #KnowledgeBase
---
# «Статус монетизации партнеров в день» — t_partners_monetization_status

## Названия таблицы

**Короткое имя (человекочитаемое):** Статус монетизации партнеров в день  
**Тех. имя:** `t_partners_monetization_status`

## Назначение
Ежедневная статистика блокировки монетизации у партнеров по точкам установки, включая процент заблокированных точек и детализацию по городам. Содержит агрегированные данные о количестве точек установки с заблокированной и разрешенной монетизацией.

**Терминология:** под «точками установки» понимаются объекты (домофоны/камеры), у которых может быть включена или выключена монетизация.


## Хранилище и движок
- БД: `default`
- Таблица: `t_partners_monetization_status`
- Движок: `MergeTree`
- Сортировка: `ORDER BY report_date`

## DDL
```sql
CREATE TABLE t_partners_monetization_status
(
    `report_date` Date,
    `partner_uuid` String,
    `company_name` String,
    `partner_lk` String,
    `tin` String,
    `kpp` String,
    `city` String,
    `city_uuid` String,
    `total_installation_points` UInt64,
    `blocked_monetization_points` UInt64,
    `allowed_monetization_points` UInt64,
    `blocked_percentage` Float64
)
ENGINE = MergeTree()
ORDER BY report_date
```

## Поля и алиасы
| Поле                           | Тип      | Алиас (человекочитаемое)                      |
| ------------------------------ | -------- | --------------------------------------------- |
| `report_date`                  | `Date`   | `дата отчёта`                                 |
| `partner_uuid`                 | `String` | `UUID партнера`                               |
| `company_name`                 | `String` | `название компании`                           |
| `partner_lk`                   | `String` | `личный кабинет`                              |
| `tin`                          | `String` | `ИНН`                                         |
| `kpp`                          | `String` | `КПП`                                         |
| `city`                         | `String` | `название города`                             |
| `city_uuid`                    | `String` | `UUID города`                                 |
| `total_installation_points`    | `UInt64` | `всего точек установки`                       |
| `blocked_monetization_points`  | `UInt64` | `точек с заблокированной монетизацией`        |
| `allowed_monetization_points`  | `UInt64` | `точек с разрешенной монетизацией`            |
| `blocked_percentage`           | `Float64`| `процент заблокированных точек`               |

## Примечания

- **Примечание к монетизации:** поле `monetization` в исходных данных принимает значения 0 (заблокировано) и 1 (разрешено).
- **Примечание по дате.** Поле даты в этой таблице — `report_date`. Используйте его в фильтрах/агрегациях (`WHERE`, `GROUP BY`, `max(report_date)`).
- **Примечание к расчету процента:** `blocked_percentage` рассчитывается как `(заблокированные_точки * 100.0 / общее_количество_точек)` с округлением до 2 знаков после запятой.

## Частые срезы/фильтры
- По дате: `report_date`
- По партнёру: `partner_uuid`
- По городу: `city`
- По компании: `company_name`
- По уровню блокировки: `blocked_percentage`
- По количеству точек: `total_installation_points`

## Примеры запросов (с алиасами)

1) Статус монетизации за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_partners_monetization_status
)
SELECT
  t.partner_uuid AS `UUID партнера`,
  t.company_name AS `название компании`,
  t.total_installation_points AS `всего точек установки`,
  t.blocked_monetization_points AS `точек с заблокированной монетизацией`,
  t.blocked_percentage AS `процент заблокированных точек`
FROM t_partners_monetization_status AS t
INNER JOIN max_dt USING(report_date);
```

2) Топ-10 партнеров с наибольшим процентом заблокированной монетизации:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_partners_monetization_status
)
SELECT
  t.company_name AS `название компании`,
  t.city AS `название города`,
  t.blocked_percentage AS `процент заблокированных точек`,
  t.total_installation_points AS `всего точек установки`
FROM t_partners_monetization_status AS t
INNER JOIN max_dt USING(report_date)
WHERE t.total_installation_points > 0
ORDER BY t.blocked_percentage DESC
LIMIT 10;
```

3) Статистика по городам за последнюю дату:
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_partners_monetization_status
)
SELECT
  t.city AS `название города`,
  COUNT(*) AS `количество партнеров`,
  SUM(t.total_installation_points) AS `общее количество точек`,
  SUM(t.blocked_monetization_points) AS `общее количество заблокированных точек`,
  ROUND(AVG(t.blocked_percentage), 2) AS `средний процент блокировки`
FROM t_partners_monetization_status AS t
INNER JOIN max_dt USING(report_date)
WHERE t.city != ''
GROUP BY t.city
ORDER BY 4 DESC;
```

4) Партнеры с полной блокировкой монетизации (100%):
```sql
WITH max_dt AS (
  SELECT max(report_date) AS report_date
  FROM t_partners_monetization_status
)
SELECT
  t.partner_uuid AS `UUID партнера`,
  t.company_name AS `название компании`,
  t.city AS `название города`,
  t.total_installation_points AS `всего точек установки`
FROM t_partners_monetization_status AS t
INNER JOIN max_dt USING(report_date)
WHERE t.blocked_percentage = 100.0
ORDER BY t.total_installation_points DESC;
```

## Ключи и соединения
- Сортировка `ORDER BY report_date`. Используйте `report_date` для фильтров по дате; `partner_uuid` для группировок по партнерам; `city` для городских срезов.
- Связь с таблицами: [[installation_point_st_partner]], [[companies_dir_partner]], [[entries_installation_points_dir_partner]]

## Ограничения и примечания
- Данные — **за день** (ежедневный снимок). Для анализа трендов используйте агрегацию по периодам.
- NULL-значения в полях `company_name`, `partner_lk`, `tin`, `kpp`, `city`, `city_uuid` заменяются на пустые строки.
- Процент блокировки рассчитывается только для партнеров с ненулевым количеством точек установки.
- Данные формируются через материализованное представление с ежедневным обновлением в 05:33.
