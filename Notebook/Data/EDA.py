#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# ### This project understand how the student's performence is affected by other veriables

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Import CSV dataset file 

# In[2]:


df = pd.read_csv('stud.csv')
df.head(5)


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# ### There are no missing in dataset 

# In[6]:


df.duplicated().sum()


# ### There are no duplicate vaue in dataset

# In[7]:


df.describe()


# - All means are very close to each other 
# - All standerd deviations also very close to each other

# In[8]:


df.nunique()


# In[9]:


print("Gender:",df['gender'].unique())

print("race_ethnicity:",df['race_ethnicity'].unique())

print("parental_level_of_education:",df['parental_level_of_education'].unique())

print("lunch:",df['lunch'].unique())

print("test_preparation_course:",df['test_preparation_course'].unique())


# ## we can add 2 colums in dataset which is total marks and average

# In[10]:


df["total_score"]=df["math_score"]+df["reading_score"]+df["writing_score"]
df["average"]= df["total_score"]/3
df.head()


# ## Let's see some plots

# - Female students performence better than male students

# In[11]:


fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',kde=True,hue='gender')
plt.show()


# In[12]:




plt.subplots(1,3,figsize=(25,10))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='race_ethnicity')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race_ethnicity')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race_ethnicity')
plt.show()


# - students group A and E performence is very poor whether the are female or male 

# In[13]:



plt.subplots(1,3,figsize=(25,10))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='parental_level_of_education')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='parental_level_of_education')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='parental_level_of_education')
plt.show()


# - 2nd plot shows that parent's whose education is of associate's degree or master's degree their male's performance is well
# - 3rd plot we can see there is no effect of parent's education on female students.

# In[14]:


plt.subplots(1,3,figsize=(25,10))
plt.subplot(141)
ax =sns.histplot(data=df,x='average',kde=True,hue='lunch')
plt.subplot(142)
ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
plt.subplot(143)
ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
plt.show()


# - Standard lunch helps perform well in exams.
# - Standard lunch helps perform well in exams be it a male or a female.

# In[15]:




plt.figure(figsize=(18,8))
plt.subplot(1, 4, 1)
plt.title('MATH SCORES')
sns.violinplot(y='math_score',data=df,color='green',linewidth=5)
plt.subplot(1, 4, 2)
plt.title('READING SCORES')
sns.violinplot(y='reading_score',data=df,color='yellow',linewidth=5)
plt.subplot(1, 4, 3)
plt.title('WRITING SCORES')
sns.violinplot(y='writing_score',data=df,color='red',linewidth=5)
plt.show()


# - its clearly visible that most of the students score in between 60-80

# In[28]:


plt.subplot(1, 5, 1)
size = df['gender'].value_counts()
labels = 'Female', 'Male'
color = ['yellow','red']


plt.pie(size, colors = color, labels = labels,autopct = '.%2f%%',radius=5)
plt.title('Gender', fontsize = 20)
plt.show()


# - Male and Female almost same

# In[29]:


plt.subplot(1, 5, 2)
size = df['race_ethnicity'].value_counts()
labels = 'Group C', 'Group D','Group B','Group E','Group A'
color = ['red', 'yellow', 'blue', 'green','orange']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%',radius=5)
plt.title('race_ethnicity', fontsize = 20)
plt.show()


# - Group C has most number of students

# In[30]:


plt.subplot(1, 5, 3)
size = df['lunch'].value_counts()
labels = 'Standard', 'Free'
color = ['yellow','green']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%',radius=5)
plt.title('Lunch', fontsize = 20)
plt.show()


# - almost 65% student prefer standerd lunch

# In[35]:


plt.subplot(1, 5, 4)
size = df['test_preparation_course'].value_counts()
labels = 'None', 'Completed'
color = ['yellow','green']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%',radius=5)
plt.title('Test Course', fontsize = 15)
plt.show()


# - 64% student not completed test preparation course

# In[34]:


plt.subplot(1, 5, 5)
size = df['parental_level_of_education'].value_counts()
labels = 'Some College', "Associate's Degree",'High School','Some High School',"Bachelor's Degree","Master's Degree"
color = ['red', 'green', 'blue', 'cyan','orange','grey']

plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%',radius=5)
plt.show()


# - Number of students whose parental education is "Some College" and "Associate's Degree" is greater 

# In[41]:




gender_group = df.groupby('gender').mean()
gender_group



plt.figure(figsize=(10, 8))

X = ['Total Average','Math Average']


female_scores = [gender_group['average'][0], gender_group['math_score'][0]]
male_scores = [gender_group['average'][1], gender_group['math_score'][1]]

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, male_scores, 0.4, label = 'Male')
plt.bar(X_axis + 0.2, female_scores, 0.4, label = 'Female')
  
plt.xticks(X_axis, X)
plt.ylabel("Marks")
plt.title("Total average v/s Math average marks of both the genders", fontweight='bold')
plt.legend()
plt.show()


# 
#    - On an average females have a better overall score than men.
#    - whereas males have scored higher in Maths.
# 

# In[43]:


plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')
sns.countplot(df['parental_level_of_education'], palette = 'Blues')
plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()


# - largest number of parents are from some college

# In[46]:




df.groupby('parental_level_of_education').agg('mean').plot(kind='barh',figsize=(10,10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# - The score of student whose parents possess master and bachelor level education are higher than others.

# In[48]:


plt.subplots(1,4,figsize=(16,5))
plt.subplot(141)
sns.boxplot(df['math_score'],color='skyblue')
plt.subplot(142)
sns.boxplot(df['reading_score'],color='hotpink')
plt.subplot(143)
sns.boxplot(df['writing_score'],color='yellow')
plt.subplot(144)
sns.boxplot(df['average'],color='lightgreen')
plt.show()


# - Outliers 

# In[49]:


sns.pairplot(df,hue = 'gender')
plt.show()


# - From the above plot it is clear that all the scores increase linearly with each other.

# In[ ]:




