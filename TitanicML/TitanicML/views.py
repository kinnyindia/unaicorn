from django.shortcuts import render
from . import fakemodel
from . import MLPredict


def home(request):
    return render(request, 'index.html')


def result(request):
    pclass = request.GET['pclass']
    sex = request.GET['sex']
    age = request.GET['age']
    sibsp = request.GET['sibsp']
    parch = request.GET['parch']
    fare = request.GET['fare']
    embarked = request.GET['embarked']
    title = request.GET['title']
    # prediction = fakemodel.fakepredict(int(userinputage))
    prediction = MLPredict.predictionmodel(
        pclass, sex, age, sibsp, parch, fare, embarked, title)
    return render(request, 'result.html', {'prediction': prediction})
