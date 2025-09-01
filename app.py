from flask import Flask, request, jsonify, Response, render_template, redirect, url_for
from process_request import *
from helper import is_file_allowed

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file is None or file.filename == '':
        return redirect(request.url)

    if is_file_allowed(file):
        return process_image(file)

    return redirect(request.url)


@app.route('/upload_video', methods=['POST'])
def video():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file is None or file.filename == '':
        return redirect(request.url)

    if is_file_allowed(file):
        return process_video(file)

    return redirect(request.url)


@app.route('/realtime')
def realtime():
    return render_template('realtime_feed.html')


@app.route('/api/realtime_feed')
def realtime_feed():
    return Response(generate_realtime_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def history():
    pass


if __name__ == '__main__':
    app.run()
