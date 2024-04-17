# from server import app

# if __name__ == "__main__":
#         app.run()

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def slideshow():
    images = [
        '/dataset/AIC2023/pumkin_dataset/0/frames/pyscenedetect/Keyframes_L01/keyframes/L01_V001/00000.jpg',
        '/dataset/AIC2023/pumkin_dataset/0/frames/pyscenedetect/Keyframes_L01/keyframes/L01_V001/00032.jpg',
        '/dataset/AIC2023/pumkin_dataset/0/frames/pyscenedetect/Keyframes_L01/keyframes/L01_V001/00047.jpg',
    ]
    return render_template('slide_show.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)

