import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from lightgbm import LGBMRegressor
import joblib
#from catboost import CatBoostRegressor

# Load your dataset
# Assuming your dataset has 'text_data' and 'personality_score' columns
# df = pd.read_csv('your_dataset.csv')

# Assuming you have already preprocessed your text data and stored it in 'text_data'
text_data = training_data['text_data']
personality='neu'
personality_scores = training_data[personality]

vectorizer = TfidfVectorizer(ngram_range=(1,1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, personality_scores, test_size=0.2, random_state=42)
X_train_tfidf = vectorizer.fit_transform(text_data)
y_train=personality_scores
# Vectorize the text data using TF-IDF
scaler=StandardScaler(with_mean=False)
X_test_tfidf = vectorizer.transform(X_test)

#vectorizer=CountVectorizer()
#X_train_count = scaler.fit_transform(vectorizer.fit_transform(X_train))
#X_test_count = scaler.transform(vectorizer.transform(X_test))
# Alternatively, you can use Count Vectorizer
# vectorizer = CountVectorizer(max_features=5000)
# X_train_count = vectorizer.fit_transform(X_train)
# X_test_count = vectorizer.transform(X_test)

# List of models to try
models = [
    SVR(kernel='linear'),
    SVR(kernel='poly'),
    SVR(kernel='rbf'),
    SVR(kernel='sigmoid'),
    #CatBoostRegressor(silent=True)
]

# Train and evaluate each model
min_rmse_classifiers=[]
counter=0
for model in models:
    try:
      model.fit(X_train_tfidf, y_train)
    except:
      continue
    joblib.dump(model,"/content/drive/MyDrive/CAPSTONE_19BCE1180"+"/"+model.__class__.__name__+"_"+personality+"_"+str(counter)+".pkl")
    counter+=1
    y_pred = model.predict(X_test_tfidf)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model.__class__.__name__} - Root Mean Squared Error: {mse**0.5}")



text_data = training_data['text_data']
vectorizer = TfidfVectorizer(ngram_range=(1,1))
X_train_tfidf = vectorizer.fit_transform(text_data)

y_train2=training_data["neu"]
y_train3=training_data["ext"]
y_train4=training_data["agr"]
y_train5=training_data["con"]

#neu
svr_neu=SVR(kernel='poly')
svr_neu.fit(X_train_tfidf, y_train2)
joblib.dump(svr_neu,"/content/drive/MyDrive/CAPSTONE_19BCE1180"+"/"+svr_neu.__class__.__name__+"_"+"neu"+"_"+"1"+".pkl")
