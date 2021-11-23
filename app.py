from flask import Flask, request, render_template, Response
from flask_cors import CORS
from web_app import game1, game2, game3, game4, score_list

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/game1')
def game_page1():
    return render_template('game1.html')


@app.route('/game2')
def game_page2():
    return render_template('game2.html')


@app.route('/game3')
def game_page3():
    return render_template('game3.html')


@app.route('/game4')
def game_page4():
    return render_template('game4.html')


@app.route('/result')
def result():
    return render_template('result.html', data=score_list)


@app.route('/game_1_video_feed')
def game_1_video_feed():
    return Response(game1(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/game_2_video_feed')
def game_2_video_feed():
    return Response(game2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/game_3_video_feed')
def game_3_video_feed():
    return Response(game3(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/game_4_video_feed')
def game_4_video_feed():
    return Response(game4(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)
