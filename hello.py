from flask import Flask,request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# @app.route("/niladri")
# def hello_world2():
#     return "<p>Hello, niladri!</p>"


# @app.route("/user/<user_name>")
# def user_defined(user_name):
#     return f"Hello {escape(user_name)}"

@app.route('/user/<username>')
def show_user_profile(username):
    return f'User {username}'


@app.route('/sum/<x>/<y>')
def sum_two_num(x,y):
    return f'Sum of {x} and {y} = {int(x) + int(y)}'

@app.route('/sum_numbers', methods = ['POST'])
def sum_num():
    data = request.get_json( )
    x = data['x']
    y = data['y']
    sum = int(x) + int(y)
    print(sum)
    return str(sum)