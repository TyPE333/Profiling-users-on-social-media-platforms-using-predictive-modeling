import sys
sys.path.insert(0,'/path/to/mod_directory')
import numpy as np
import pandas as pd
import sklearn
import xml.etree.ElementTree as ET
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
#import xgboost
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

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


test_profile_data=pd.read_csv(input_data_path + "profile/profile.csv",index_col=0)
text_data = {}  # Dictionary to store user IDs and text data
text_folder = input_data_path+'text'
user_ids = test_profile_data['userid']
for user_id in user_ids:
    file_path = os.path.join(text_folder, f'{user_id}.txt')
    #print(file_path)
    if os.path.exists(file_path):
        try:
          with open(file_path, 'r',encoding='utf-8',errors='ignore') as file:
            text_data[user_id] = file.read()
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: Skipping file for user ID {user_id}")
            text_data[user_id] = None  # Handle unreadable files
    else:
        text_data[user_id] = None  # Handle missing files if needed


test_profile_data['text_data'] = [text_data[user_id] for user_id in test_profile_data['userid']]

training_data=pd.read_csv("/home/itadmin/Downloads/text_classification_data.csv",index_col=0)
text_training_data=training_data[["userid","text_data"]]

vectorizer1 = TfidfVectorizer(ngram_range = (1,1))
train_vectors=normalize(vectorizer1.fit_transform(text_training_data["text_data"]))


X_test_tfidf=normalize(vectorizer1.transform(test_profile_data["text_data"]))

vectorizer2 = TfidfVectorizer(ngram_range = (2,2),max_features=5000)
train_vectors=normalize(vectorizer2.fit_transform(text_training_data["text_data"]))


X_test_tfidf_svr=normalize(vectorizer2.transform(test_profile_data["text_data"]))


#age interval classification using svm
text_data=pd.read_csv("/home/itadmin/Downloads/text_classification_data.csv",index_col=0)
text_data=text_data[["userid","text_data","age"]]
age_group_list=[]
for i in text_data.age:
  if 0<=i<=24:
    age_group_list.append("xx-24")
  elif 25<=i<=34:
    age_group_list.append("25-34")
  elif 35<=i<=49:
    age_group_list.append("35-49")
  else:
    age_group_list.append("50-xx")
age_group_list=pd.Series(age_group_list)

text_data.insert(2,"age_group",age_group_list)

vectorizer3 = TfidfVectorizer(ngram_range = (1,1))
train_vectors=vectorizer3.fit_transform(text_data["text_data"])
test_vectors=vectorizer3.transform(test_profile_data["text_data"])
sv_age=joblib.load("/home/itadmin/Downloads/svm_age_pred.pkl")
age_preds=sv_age.predict(test_vectors)

#personality prediction using text data
svr_ope=joblib.load("/home/itadmin/Downloads/SVR_ope_2.pkl")
svr_neu=joblib.load("/home/itadmin/Downloads/SVR_neu_1(1).pkl")
svr_ext=joblib.load("/home/itadmin/Downloads/SVR_ext_2(1).pkl")
svr_agr=joblib.load("/home/itadmin/Downloads/SVR_agr_1(1).pkl")
svr_con=joblib.load("/home/itadmin/Downloads/SVR_con_1(2).pkl")

ope_predict=svr_ope.predict(test_vectors)
neu_predict=svr_neu.predict(test_vectors)
ext_predict=svr_ext.predict(test_vectors)
agr_predict=svr_agr.predict(test_vectors)
con_predict=svr_con.predict(test_vectors)

## gender prediction using text data
sv_classifier=joblib.load("/home/itadmin/Downloads/svclassifier.pkl")
predictions=sv_classifier.predict(X_test_tfidf)

k=[]
for i in range(0,len(predictions)):
  if predictions[i]==0:
    k.append("male")
  else:
    k.append("female")

## age prediction using text data
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf

# Load the saved tokenizer
#saved_tokenizer = DistilBertTokenizer.from_pretrained("/home/itadmin/Downloads/model_tokenizer/f")
'''
# Load the saved DistilBERT model
saved_tokenizer = AutoTokenizer.from_pretrained("/home/itadmin/Downloads/model_tokenizer/tokenizer")


saved_model = tf.keras.models.load_model("/home/itadmin/Downloads/model_tokenizer/model")



# Unseen test data
unseen_test_data = list(test_profile_data["text_data"])
# Tokenize the unseen test data
inputs = saved_tokenizer(unseen_test_data, return_tensors='tf', truncation=True, padding=256, max_length=256, add_special_tokens=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Make predictions
predictions = saved_model.predict([input_ids, attention_mask])

# Extract predicted labels
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Print or use the predicted_labels as needed
print(predicted_labels)
'''
'''
### PREDICTING PERSONALITY SCORES USING LIWC FEATURES
training_profile_data=pd.read_csv("/data/training/profile/profile.csv",index_col=0)
training_profile_data.rename(columns={"userid":"userId"},inplace=True)
training_liwc_features=pd.read_csv("/data/training/LIWC/LIWC.csv")
temp1=training_liwc_features.drop("Seg",axis=1)
set1 = temp1.iloc[:, :81]
set2 = training_profile_data.iloc[:,1:8]
merged=pd.merge(set1,training_profile_data,on="userId")
set1 = merged.iloc[:, 1:81]
set2 = merged.iloc[:,83:]

X_train= set1  
# Fit the scaler on your data and transform it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train = set2["agr"]



test_profile_data.rename(columns={"userid":"userId"},inplace=True)
test_liwc_path= input_data_path+'LIWC/LIWC.csv'
test_liwc_features=pd.read_csv(test_liwc_path)
temp2=test_liwc_features.drop("Seg",axis=1)
set3 = temp2.iloc[:, :81]
merged=pd.merge(set3,test_profile_data,on="userId")
set3 = merged.iloc[:, 1:81]
X_test = set3
X_test_scaled=scaler.transform(X_test)

xgb_agr=joblib.load("/home/itadmin/Downloads/xgb_agr.pkl")
agr_preds=xgb_agr.predict(X_test_scaled)

'''


# training_data_stats
counter=0
for x in test_profile_data.userid:
    output_xml_file_name = output_data_path + str(x)+".xml"
    y=k[counter]
    ope=str(round(ope_predict[counter],6))
    neu=str(round(neu_predict[counter],6))
    ext=str(round(ext_predict[counter],6))
    agr=str(round(agr_predict[counter],6))
    con=str(round(con_predict[counter],6))
    
    training_data_stats={
    "age_group": age_preds[counter],
    "gender": y,
    "extrovert": ext,
    "neurotic":neu ,
    "agreeable":agr,
    "conscientious":con,
    "open": ope

	}
    counter=counter+1

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
