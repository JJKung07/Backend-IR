# Project Setup

This guide will walk you through setting up and running the project.

## Prerequisites
Ensure you have the following installed on your system:
- Python (>= 3.x)
- SQLite (or any required database system)
- Required dependencies (can be installed using `pip`)

## Setup Instructions

### 1. Create a Resources Directory
Create a directory named `Resources` in the project root:
```sh
mkdir Resources
```

### 2. Create a Database Inside `Resources`
Navigate to the `Resources` directory and create a database file:
```sh
cd Resources
sqlite3 database.db "VACUUM;"
cd ..
```
Alternatively, if you are using a different database system, set up the required configurations accordingly.

### 3. Run `SQL.py`
Execute the SQL script to set up the database schema:
```sh
python SQL.py
```
This will initialize the required tables and data inside `database.db`.

### 4. Run `app.py`
Start the application:
```sh
python app.py
```
This will launch the application and start serving requests.

## File Structure
```
project-root/
├── Resources/
│   ├── database.db
├── SQL.py
├── app.py
├── requirements.txt
├── README.md
```

