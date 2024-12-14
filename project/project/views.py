from django.shortcuts import render

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression


def user(request):
    return render(request,'userinput.html')

def viewdata(request):
    df = pd.read_csv("C:/Users/PMLS/Documents/ML/ML Algorithms/heart.csv")
    x = df.drop(["target"] , axis = 1)
    y = df["target"]
    X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 41)
    
    LR = LogisticRegression()      
    LR.fit(X_train , y_train)
    
  # Gather user input into a single list (ensure feature order matches training data)
    new_data = [
        int(request.GET['age']),
        int(request.GET['sex']),
        int(request.GET['cp']),
        int(request.GET['trestbps']),
        int(request.GET['chol']),
        int(request.GET['fbs']),
        int(request.GET['restecg']),
        int(request.GET['thalach']),
        int(request.GET['exang']),
        float(request.GET['oldpeak']),
        int(request.GET['slope']),
        int(request.GET['ca']),
        int(request.GET['thal']),
    ]
  
    y_pred = LR.predict([new_data])
   
    data = {
            'prediction': y_pred[0],
            'message': '',
           }

    if y_pred == 0:
        data['message'] = 'The heart is predicted to have no disease'
        
        
    else:
          data['message'] = 'The heart is predicted to have disease'

        
    return render(request,'viewdata.html' , data)