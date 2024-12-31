from flask import Flask, request, render_template_string

app = Flask(__name__)

# HTML Template
html_template = """
<!doctype html>
<html>
    <head>
        <title>Multiplication Table</title>
    </head>
    <body>
        <h1>Enter a Number</h1>
        <form method="POST">
            <input type="number" name="number" required>
            <button type="submit">Get Table</button>
        </form>
        {% if table %}
        <h2>Table for {{ number }}:</h2>
        <ul>
            {% for row in table %}
            <li>{{ row }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def table():
    table = None
    number = None
    if request.method == 'POST':
        try:
            number = int(request.form['number'])
            table = [f"{number} x {i} = {number * i}" for i in range(1, 11)]
        except ValueError:
            table = ["Invalid input. Please enter a valid number."]

    return render_template_string(html_template, table=table, number=number)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)