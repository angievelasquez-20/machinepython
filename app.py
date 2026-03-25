from flask import Flask , render_template, request
import LinearRegression
import LogisticRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/FirstPage')
def fisrtPage():
    return render_template('caso2.html')

@app.route('/secondPage')
def actividadE():
    return render_template('caso1.html')

@app.route('/thirdPage')
def caso3():
    return render_template('caso3.html')

@app.route('/linearRegresion/', methods = ["GET","POST"])
def calculatorGrade():
    calculateResult = None 
    plot_url = None
    if request.method == "POST":
        hours = float (request.form['hours'])
        calculateResult = LinearRegression.calculatorGrade(hours)
        LinearRegression.generate_plot(hours, calculateResult)
        plot_url = 'plot.png'
    return render_template('LinearRegressionGrades.html', result = calculateResult, plot_url=plot_url)

@app.route('/logisticRegression/', methods=["GET", "POST"])
def predictPurchase():
    calculateResult = None
    plot_url = None
    
    if request.method == "POST":
        age = float(request.form['age'])
        income = float(request.form['income'])
        visits = float(request.form['visits'])
        time = float(request.form['time'])
        purchases = float(request.form['purchases'])
        discount = float(request.form['discount'])

        prob = LogisticRegression.predict_purchase(age, income, visits, time, purchases, discount)
        
        if prob > 0.5:
            calculateResult = f"Will Purchase (Probability: {prob:.2f})"
        else:
            calculateResult = f"Will Not Purchase (Probability: {prob:.2f})"
        
        LogisticRegression.generate_plot(age, income, visits, time, purchases, discount)
        plot_url = 'logistic_plot.png'

    return render_template('LogisticRegression.html', result=calculateResult, plot_url=plot_url)