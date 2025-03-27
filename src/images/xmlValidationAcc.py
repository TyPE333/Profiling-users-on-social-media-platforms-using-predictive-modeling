# -*- coding: utf-8 -*-
"""
XML Validation Script for Gender Classification
TCSS 551 Machine Learning
@author: Carla Peterson
Created on Tue Nov 21 21:31:02 2023
xmlValidationAcc.py

This Python script was developed to evaluate accuracy of the CNN image model
on test data against the ground truth. 
"""

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd

# Check if two command-line arguments were provided
if len(sys.argv) != 3:
    print("Usage: xmlValidationAcc.py <xml_dir_path> <profile_file_path>")
    sys.exit(1)

# Retrieve the input values from command-line arguments
xml_data_path= sys.argv[1]
profile_data_path= sys.argv[2]

#input_data_path = "D:/DOCUMENTS/Python/TCSS555-Project/tcss555/public-test-data/output/"
# input_data_path = "D:/DOCUMENTS/Python/TCSS555-Project/tcss555/training/output/"
profile_df = pd.read_csv(profile_data_path)

xml_filenames = os.listdir(xml_data_path)
female_num = 0
male_num = 0
tp = 0
fp = 0

for xml_filename in xml_filenames:
    tree = ET.parse(os.path.join(xml_data_path + xml_filename))
    root = tree.getroot()
    userid = root.get('userid')
    gender_value = profile_df.loc[profile_df['userid'] == userid, 'gender'].values[0]
    xml_gender_val = root.get('gender')
    if gender_value == '0' & xml_gender_val == "male":
        tp = tp + 1
    
    elif gender_value == '1' & xml_gender_val == "female":
        tp = tp + 1
    else:
        fp = fp + 1
        
print("Number of true positives: " + str(tp))
print("Number of false positives: " + str(fp))
                    
