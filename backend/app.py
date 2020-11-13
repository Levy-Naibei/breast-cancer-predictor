from flask import Flask, render_template, request

# initiatialize flask app
app = Flask(__name__)

# home page
@app.route('/')
def defaultpage():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)