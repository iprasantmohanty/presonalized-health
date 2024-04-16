import sqlite3

##connection
connection=sqlite3.connect("multi_diseases.db")

##create curoser
cursor=connection.cursor()


## create the table
table_info="""
create table if not exists DISEASE (FEATURES VARCHAR(25),option1 VARCHAR(10),option2 VARCHAR(10),option3 VARCHAR(10),option4 VARCHAR(10),option5 VARCHAR(10));
"""

cursor.execute(table_info)

## Insert Some more records
##"""
cursor.execute('''Insert Into DISEASE (FEATURES) values('age')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2) values('have hypertension ?','yes','no')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2) values('have heart Disease ?','yes','no')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2) values('gender','male','female')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2) values('ever married ?','yes','no')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2,option3,option4,option5) values('work type','govt_job','never worked','private','self-employed','children')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2) values('residence type ','rural','urban')''')
cursor.execute('''Insert Into DISEASE (FEATURES,option1,option2,option3,option4) values('smoking status','unknown','formaly smoked','never smoked','smokes')''')
##"""
##cursor.execute('''DELETE FROM DISEASE''')
## Disspaly all the records

print("The inserted records are")
data=cursor.execute('''Select * from DISEASE''')
for row in data:
    print(row)

## Commit your changes int he databse
connection.commit()
connection.close()