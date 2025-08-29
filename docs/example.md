# Пример базы знаний

Этот документ используется для теста поиска в ChromaDB.

## Таблица: users

```sql
CREATE TABLE users
(
    user_id UInt32,
    username String,
    signup_date Date,
    last_login DateTime
)
ENGINE = MergeTree()
ORDER BY user_id
