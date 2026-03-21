from flask import Flask , render_template, request
import LinearRegression

app = Flask(__name__)

# Función de predicción sencilla (reemplaza esto por un modelo real si lo tienes)
def predict_purchase(age, income, visits, time, purchases, discount):
    score = age * 0.05 + income * 0.01 + visits * 0.2 + time * 0.1 + purchases * 0.4 - discount * 0.3
    return 1 if score >= 10 else 0

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/FirstPage')
def fisrtPage():
    return render_template('index.html')

@app.route('/secondPage')
def actividadE():
    return render_template('actividadE.html')

@app.route('/linearRegresion/', methods = ["GET","POST"])
def calculatorGrade():
    calculateResult = None 
    if request.method == "POST":
        hours = float (request.form['hours'])
        calculateResult = LinearRegression.calculatorGrade(hours)
    return render_template('LinearRegressionGrades.html', result = calculateResult)

@app.route('/logisticRegression/', methods=["GET", "POST"])
def predictPurchase():
    calculateResult = None
    
    if request.method == "POST":
        age = float(request.form['age'])
        income = float(request.form['income'])
        visits = float(request.form['visits'])
        time = float(request.form['time'])
        purchases = float(request.form['purchases'])
        discount = float(request.form['discount'])

        prediction = predict_purchase(age, income, visits, time, purchases, discount)
        
        if prediction == 1:
            calculateResult = "Will Purchase"
        else:
            calculateResult = "Will Not Purchase"

    return render_template('LogisticRegression.html', result=calculateResult)