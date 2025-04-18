# -*- coding: utf-8 -*-
"""MLNB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/150TEbze1W8gK0FvQv1F1Z_wPoOUKKW7I
"""

import sys
sys.path.insert(0,'/path/to/mod_directory')
import numpy as np
import pandas as pd
import sklearn
import xml.etree.ElementTree as ET
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib
# Check if two command-line arguments were provided
if len(sys.argv) != 3:
    print("Usage: mymodel.py <input1> <input2>")
    sys.exit(1)

# Retrieve the input values from command-line arguments
input_data_path= sys.argv[1]
output_data_path= sys.argv[2]

# Print the inputs
print("Input 1:",input_data_path)
print("Input 2:",output_data_path)
#/content/drive/MyDrive/SEM 6/tcss555/public-test-data/relation/relation.csv
# Read the input profile data
test_profile_data=pd.read_csv(input_data_path + "profile/profile.csv")
test_relations_data=pd.read_csv(input_data_path + "relation/relation.csv")
test_profile_data.drop("Unnamed: 0",inplace=True,axis=1)
test_relations_data.drop("Unnamed: 0",inplace=True,axis=1)
test_profile_data=test_profile_data[["userid","gender"]]
result = test_relations_data.groupby('userid')['like_id'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

result.rename(columns={'like_id': 'likes'}, inplace=True)
test_gender_rel=pd.merge(result,test_profile_data,on="userid")

gender_data=pd.read_csv("/data/training/profile/profile.csv")
relation_data=pd.read_csv("/data/training/relation/relation.csv")
gender_data.drop("Unnamed: 0",inplace=True,axis=1)
relation_data.drop("Unnamed: 0",inplace=True,axis=1)
gender_data=gender_data[["userid","gender"]]

result = relation_data.groupby('userid')['like_id'].apply(lambda x: ' '.join(x.astype(str))).reset_index()

result.rename(columns={'like_id': 'likes'}, inplace=True)
gender_rel=pd.merge(result,gender_data,on="userid")

Vectorizer=CountVectorizer()
New_X_train=Vectorizer.fit_transform(gender_rel["likes"])
X_test_counts=Vectorizer.transform(test_gender_rel["likes"])
clf = joblib.load('/home/itadmin/Downloads/multinomial_nb_model.pkl')
predictions=clf.predict(X_test_counts)
k=[]
for i in range(0,len(predictions)):
  if predictions[i]==0:
    k.append("male")
  else:
    k.append("female")


# training_data_stats
counter=0
for x in test_gender_rel.userid:
    output_xml_file_name = output_data_path + str(x)+".xml"
    y=k[counter]
    counter=counter+1
    training_data_stats={
    "age_group": "xx-24",
    "gender": y,
    "extrovert": "3.49",
    "neurotic": "2.73",
    "agreeable": "3.58",
    "conscientious": "3.45",
    "open": "3.91"

	}


    temp={"id":str(x)}
    temp.update(training_data_stats)
    root = ET.Element("user",temp)
    tree = ET.ElementTree(root)
    tree.write(output_xml_file_name)

# Create the root element
# root = ET.Element("root")

# # Create child elements
# child1 = ET.SubElement(root, "child1")
# child2 = ET.SubElement(root, "child2")

# # Add some data to the child elements
# child1.text = "Data for Child 1"
# child2.text = "Data for Child 2"

# # Create an XML tree
# tree = ET.ElementTree(root)

# # Save the XML to a file
# tree.write(output_data_path+"/"+"/output_file.xml", encoding="utf-8", xml_declaration=True)

# print("XML file 'output.xml' has been created.")