# PyMongo
- A Python distribution containing tools for working with MongoDB

# Installation
```
pip install pymongo
```

# How to Use
- Import the module.
```python
import pymongo
```
- Make a connection with MongoClient.
```python
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
# client = MongoClient("mongodb://username:password@localhost:27017")
```
- Get a database.
```python
db = client.test_database
# db = client["test-database"]
```
- Get a collection.
```python
collection = db.test_collection
# collection = db["test-collection"]
```
- Execute queries.
```python
# insert
post = {
    "author": "Mike",
    "text": "My first blog post!",
    "tags": ["mongodb", "python", "pymongo"],
}

posts = db.posts
posts.insert_one(post)

new_posts = [
    {
        "author": "Mike",
        "text": "Another post!",
        "tags": ["bulk", "insert"]
    },
    {
        "author": "Eliot",
        "title": "MongoDB is fun",
        "text": "and pretty easy too!"
    },
]

posts.insert_many(new_post)

# read
single_document = posts.find_one()
print(single_print) 

documents = posts.find()
for document in documents:
    print(document)

# count
posts.count_documents({})
posts.count_documents({"author": "Mike"})
```
- Close.
```python
client.close()
```
