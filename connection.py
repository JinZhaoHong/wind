import pandas as pd
from sqlalchemy import create_engine
import psycopg2

connect = psycopg2.connect(
	host="switch-db2.erg.berkeley.edu",
	database="postgres",
	user="wonho", 
	password="RAELerg_%673#",
	port="5433", 
	sslmode="require")

print("It's connected")
print("Now it's querying...")

cur = connect.cursor()
cur.execute('SELECT version()')

queried_data = cur.fetchall()

print(queried_data)


