import sklearn
import numpy as np
import pandas as pd
import matplotlib as plt

dating = pd.read_csv("/Users/spencerhall/Desktop/UdacityCode/speed_dating/dating_data.csv")
"""
print dating.head()
print "The data has", dating.shape[0], "rows and", dating.shape[1], "columns."
"""
# This dataset has a very large number of variables (195), many of which appear
# from the heading to have quite a few missing values. Before determining
# what question seems most interesting to pursue from the data, I need
# to check each column to make sure it has enough available data to be
# useful for the analysis.

# This function goes through the entire dataframe and prints out which columns
# have above 90% of their values present, between 80% and 90%, between 70% and
# 80%, between 60% and 70%, and below 60%.

def Find_NaN(arg):
    try:
        result = (arg == "NaN" or arg == None or np.isnan(arg) == True)
    except (TypeError):
        result = (arg == "NaN" or arg == None)
    return result


def Summary_of_Missing(df):
    percentage_missing = []
    rows = df.axes[0]
    for i in rows:
        temp = []
        temp = map(Find_NaN, df.loc[i])
        mean = np.mean(temp)
        percentage_missing.append(mean)
    return percentage_missing


col_names = list(dating.axes[1])

# Extracting the first survey questions from this list, the ones before "met."

def Get_Survey(arg):
    return ((("attr" in arg) or ("sinc" in arg) or ("intel" in arg) or ("fun" in arg)
            or ("amb" in arg) or ("shar" in arg)) and (("_o" in arg) == False))

mixed_col = col_names[0:col_names.index("met") + 1]
survey_names = col_names[col_names.index("met") + 1:len(col_names) + 1]

survey_from_mixed = []
for names in mixed_col:
    if Get_Survey(names) == True:
        survey_from_mixed.append(names)

total_survey_names = ["iid", "gender"] + survey_from_mixed + survey_names

# There are 6 variables in total_survey_names that weren't filtered out
# by the Get_Survey function which have different values for each partner
# row. These are just the names of the different traits (i.e., "amb," "shar,"
# etc.). In the code block below, I remove these from the list.

to_remove = ["attr", "sinc", "intel", "fun", "amb", "shar"]
for term in to_remove:
    total_survey_names.remove(term)

# print total_survey_names

# Now we have a list which contains the names of the columns which each contain
# only a single value for each of the 552 individuals in the study, regardless
# of the number of partners. For instance, if an individual had 5 partners,
# she would have five rows in the dataset. The columns in total_survey_names
# would have the same value for each row because they have only one answer
# per person. The other columns have a different answer for each partner.

# To do next:
# Find out which of the survey questions got answered the most.
# Perhaps I should just split the dataset up so that there's one smaller
# dataset (552 rows) with one row per person containing each of the
# columns in total_survey_names, and a larger dataset with information only
# on the different partner answers for each person. Then I can see which columns
# in each of the two datasets have the most recorded observations.
#
# http://stackoverflow.com/questions/11285613/selecting-columns

#### Creating the smaller, survey dataset.

blah = list(dating["iid"])
indices_list = []

# http://stackoverflow.com/questions/8293086/python-continue-iteration-of-for-loop-on-exception
for i in range(1, 513):
    try:
        indices_list.append(blah.index(i))
    except (ValueError):
        continue

survey_dataset = pd.DataFrame(dating.loc[indices_list])

#### Creating the larger dataset with information on each partner.

# Citation for .copy() and .drop():
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.copy.html
# http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.drop.html
partner_dataset = dating.copy(deep=True)
partner_dataset = partner_dataset.drop(total_survey_names, axis=1)




        





