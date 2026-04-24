from flask import Flask, render_template, request
from numpy import info
import LinearRegressionPrices as house_model
import LinearRegression
import LogisticRegression
import LogisticRegression2 as LogisticRegression
import Logisticmodel
import clustering as Clustering

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/FirstPage')
def fisrtPage():
    return render_template('caso1.html')

@app.route('/secondPage')
def actividadE():
    return render_template('caso2.html')

@app.route('/thirdPage')
def caso3():
    return render_template('caso3.html')

@app.route('/fourPage')
def caso4():
    return render_template('caso4.html')

@app.route('/supervicedLearning')
def supervicedLearning():
    return render_template('supervisedLearning.html')


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

@app.route('/GaussianNaiveBayes')
def naiveBayes():
    return render_template('GaussianNaiveBayes.html')

@app.route('/linearRegresionPrices/', methods=['GET', 'POST'])
def calories():
    result = None
    plot_url = None

    if request.method == 'POST':
        size = float(request.form['size'])
        rooms = float(request.form['rooms'])

        result = round(house_model.predict_price(size, rooms), 2)

        house_model.generate_plot(size, rooms, result)
        plot_url = "plot.png"

    return render_template('LinearRegressionPrices.html', result=result, plot_url=plot_url)

@app.route('/LogisticRegression2/', methods=["GET", "POST"])
def predictHiring():
    calculateResult = None
    plot_url = None

    if request.method == "POST":
        # Inputs from the form
        years_exp = float(request.form['years_experience'])
        python = int(request.form['skills_python'])
        sql = int(request.form['skills_sql'])
        ml = int(request.form['skills_ml'])
        dl = int(request.form['skills_deep_learning'])
        cloud = int(request.form['skills_cloud'])

        # Dictionary to hold input data for prediction
        input_data = {
            "years_experience": years_exp,
            "skills_python": python,
            "skills_sql": sql,
            "skills_ml": ml,
            "skills_deep_learning": dl,
            "skills_cloud": cloud
        }

        # Predicción
        prob = LogisticRegression.predict_hiring(input_data)

        if prob > 0.5:
            calculateResult = f"High Hiring Urgency (Probability: {prob:.2f})"
        else:
            calculateResult = f"Low Hiring Urgency (Probability: {prob:.2f})"

        # grafic generation 
        LogisticRegression.generate_confusion_matrix()
        plot_url = 'logistic2_plot.png'

    return render_template('LogisticRegression2.html', result=calculateResult, plot_url=plot_url)

@app.route('/logisticmodel', methods=['GET', 'POST'])
def logistic_page():

    result = None
    probability = None
    prob_percent = None 
    accuracy = None
    accuracy_percent = None 
    report = None

    if request.method == 'POST':
        exp = float(request.form['experience'])
        python = int(request.form['python'])
        sql = int(request.form['sql'])
        ml = int(request.form['ml'])

        probability, prediction = Logisticmodel.predict_candidate(exp, python, sql, ml)

        result = "Hired" if prediction == 1 else "Not Hired"

        accuracy, report = Logisticmodel.generate_metrics()

        accuracy_percent = round(accuracy * 100, 2)

    return render_template('Logisticmodel.html',result=result,probability=probability,prob_percent=prob_percent,
        accuracy=accuracy,accuracy_percent=accuracy_percent,report=report)

@app.route('/clustering')
def clustering():
    info = Clustering.applyClusteringKmeans()
    return render_template('clustering.html')


if __name__ == '__main__':
    app.run(debug=True)    