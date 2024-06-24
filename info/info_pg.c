sudo -u postgres psql
\c ai_draw_db; 
SELECT * FROM ai_draw_tb;
DROP TABLE ai_draw_tb;

CREATE USER ai_user;  
CREATE DATABASE ai_draw_db;

//=======================

CREATE USER grafana_user WITH PASSWORD '******';
GRANT CONNECT ON DATABASE ai_draw_db TO grafana_user;
\c ai_draw_db
GRANT USAGE ON SCHEMA public TO grafana_user;
GRANT SELECT ON TABLE ai_draw_tb TO grafana_user;


sudo systemctl restart postgresql


psql -U grafana_user -h localhost -d ai_draw_db
//=======================
GRANT SELECT, UPDATE, INSERT ON ALL TABLES IN SCHEMA public TO "ai_user";




GRANT ALL PRIVILEGES ON DATABASE ai_draw_db to ai_user;
GRANT SELECT, UPDATE, INSERT ON ALL TABLES IN SCHEMA public TO "ai_user";
=====
GRANT pg_read_all_data TO ai_user;
GRANT pg_write_all_data TO ai_user;

ALTER USER ai_user WITH PASSWORD '*****';

Теперь подключаемся к базе, к которой хотим дать доступ:
\c ai_draw_db


GRANT ALL PRIVILEGES ON ai_draw_tb TO "ai_user"; 

11. Просмотр таблицы 
SELECT * FROM ai_draw_tb;


DROP TABLE ai_draw_tb;


Добваить в таблицу Customers
Колонку Phone 

ALTER TABLE Customers
ADD Phone CHARACTER VARYING(20) NULL;

ALTER TABLE ai_draw_tb
ADD Train_Count int NULL;

UPDATE ai_draw_tb SET Train_Start = CURRENT_TIMESTAMP WHERE line_num=1;

UPDATE ai_draw_tb SET Train_Count = 1 WHERE line_num=1;

SELECT * FROM ваша_таблица WHERE ваша_колонка_времени = (SELECT MAX(ваша_колонка_времени) FROM ваша_таблица);
SELECT * FROM ai_draw_tb WHERE Train_Count = (SELECT MAX(CURRENT_TIMESTAMP) FROM ai_draw_tb);



SELECT DISTINCT ON (Train_Count) * 
FROM ai_draw_tb 
ORDER BY Train_Count, CURRENT_TIMESTAMP DESC 
LIMIT 1;


SELECT * FROM ai_draw_tb ORDER BY Train_Count DESC LIMIT 1;

Есть таблица в базе данных postgress

CREATE TABLE ai_draw_tb(
	line_num        serial,
    user_id         varchar(80),
    num_all_files           int,
    predict_count           int,
    train_Count			    int,
    train_Start             time,
    train_End               time,   
    train_Interval          interval, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

Запрос, инкрементирующий значение ячейки Train_Count

UPDATE ai_draw_tb SET Train_Count = Train_Count + 1

UPDATE ai_draw_tb
SET Train_Count = Train_Count + 1
WHERE line_num = 2;


UPDATE ai_draw_tb
SET Train_Count = Train_Count + 1
WHERE user_id = 'user';


UPDATE ai_draw_tb
SET Train_Count = 1
WHERE user_id = 'user';

UPDATE ai_draw_tb
SET Train_Count = 1;


#=========================================================


