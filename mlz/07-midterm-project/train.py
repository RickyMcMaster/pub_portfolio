# basic
import pandas as pd
import numpy as np
import pickle

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text

print('Loading and preparing data')

# Import Data
csv_loc = 'https://www.kaggle.com/HRAnalyticRepository/employee-attrition-data#:~:text=calendar_view_week-,MFG10YearTerminationData,-.csv'

df = pd.read_csv('MFG10YearTerminationData.csv', parse_dates=[ 
    'recorddate_key',
 'birthdate_key',
 'orighiredate_key',
 'terminationdate_key',])

# Removing Retirements and Layoffs
excl_emps = df.loc[df['termreason_desc'].isin(['Layoff', 'Retirement']), 'EmployeeID']
df = df[~df['EmployeeID'].isin(excl_emps)].copy()

df.columns = df.columns.str.lower().str.replace(' ', '_')


# Adding an index field for each entry in employees' records to sequentially keep track of their history
df['emp_record_idx'] = df.groupby('employeeid')['recorddate_key'].rank(method='first')


# Sorting records
df = df.copy().sort_values(['employeeid', 'recorddate_key'])

# Adding record count for each employee
df['num_records'] = df.groupby('employeeid')['emp_record_idx'].transform('max')

# Adding job groups feature
service_staff = ['Meat Cutter', 'Cashier', 'Dairy Person', 'Produce Clerk', 'Baker',
       'Shelf Stocker',]

df['job_group'] = np.select(
    [
        df['job_title'].isin(service_staff), 
        df['job_title'].str.contains('Analyst|Exec Assistant|Clerk|Recruiter|Trainer|Benefits Admin|Auditor'),
        df['job_title'].str.contains('Legal Counsel|Corporate Lawyer'),
        df['job_title'].str.contains('Manager'),
        df['job_title'].str.contains('Director'),
        df['job_title'].str.contains('VP|CEO|CHief Information Officer')
    ], 
    [
        'Service Staff', 
        'Executive',
        'Legal',
        'Manager',
        'Director',
        'C-Suite'
    ], 
    default='Unknown'
)


# Creating new dataset only showing the last row for each employee - this is effectively an aggregated view and gives the model a more concise format to work with.
df_max = df[df.emp_record_idx==df.num_records].copy()

# Adding 'resigned' field - this will be the target for the model
df_max['resigned'] = np.where(df_max['termreason_desc']=='Resignaton', 1, 0)

# Since we are given the employees' birthdates and hire date, I thought it would make sense to create new features that are more precise than just years.
df_max['los_days'] = (df_max.recorddate_key - df_max.orighiredate_key).dt.days
df_max['age_days'] = (df_max.recorddate_key - df_max.birthdate_key).dt.days
numeric = [
'los_days',
'age_days',
# 'resigned'
]


# We can also remove the date fields
df_max.drop([
    'recorddate_key',	'birthdate_key','orighiredate_key',
    'gender_full', 
    ], axis=1, inplace=True)

# Restating categorical 
categorical = ['city_name', 'department_name',
       'job_title', 'store_name', 'gender_short', 
      'status_year',  'business_unit',
       'job_group', 'resig_month']


global_resig_rate = df_max.resigned.mean()

categorical = [e for e in categorical if e not in ('status_year', 
'status','emp_record_idx','resig_month',)]

# Store name should not be treated as a numeric data type
df_max.store_name = df_max.store_name.astype('str')


# ## Training the Model

# ### Test/Val/Train Split

target = ['resigned']

df_train_full, df_test = train_test_split(df_max[numeric+categorical+target], test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

y_train = df_train.resigned.values
y_val = df_val.resigned.values
# df_train_orig = df_train.copy()

for d in [df_train, df_val, df_test]:
    del d['resigned']


def train(df, y):
    print('Training Model')
    cat = df[categorical + numeric].to_dict(orient='rows')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20)
    model.fit(X, y)

    return dv, model

dv, model = train(df_train, y_train)

print('Saving output to pickle')
with open('resig-model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)