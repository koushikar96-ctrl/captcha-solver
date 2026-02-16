from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is working fine!"

if __name__ == '__main__':
    print(">>> Flask script started <<<")
    app.run(debug=True)
