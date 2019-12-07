import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(0,10) 
y = x * 2 
#Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
#Simple Plot
plt.plot(x,y)
#plt.show()
import pandas as pd
data = pd.read_csv('university_records.csv')
print (data)
'''
test = list(data.lookup)
print( test )
print (data.loc[:,['salary','name']])
test = data.loc[:,['salary']]
print(test)
print (np.mean(test))
print (np.average(test))
name = list(data.loc[:,['name' ]])
print(name)
'''
data1= [{'Id':100, 'Name':'Suresh', 'Profession':'Developer'},{'Id':101, 'Name':'Ramesh', 'Profession':'Trainer'}]
print(data1)
for val in data1:
     key = data1
     print( "Fine" + str(key) )
