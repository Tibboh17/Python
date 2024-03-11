# PyMySQL
- A pure-Python MySQL client library

# Installation
```
!pip install PyMySQL
```

# How to Use
- Import the module.
```python
import pymysql.cursor
```
- Connect to the database.
```python
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='passwd',
                             database='db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
``` 
- Get cursor object.
```python
cursor = connection.cursor()
```  
- Execute SQL queries.
```python
sql = "INSERT INTO users (email, password) VALUES (%s, %s)"
cursor.execute(sql, ('webmaster@python.org', 'very-secret'))
    
cursor.execute("SELECT * FROM users")
data_1 = cursor.fetchall()
data_2 = cursor.fetchone()
data_3 = cursor.fetchmany(5)
```  
- Commit.
```python
connection.commit()
```
    
- Close.
```python
connection.close()
```
