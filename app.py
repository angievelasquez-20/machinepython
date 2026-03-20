from flask import Flask , render_template, request
import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/FirstPage')
def fisrtPage():
    return render_template('index.html')

@app.route('/linearRegresion/', methods = ["GET","POST"])
def calculatorGrade():
    calculateResult = None 
    if request.method == "POST":
        hours = float (request.form['hours'])
        calculateResult = LinearRegression.calculatorGrade(hours)
    return render_template('LinearRegressionGrades.html', result = calculateResult)
