from flask import Flask, request, jsonify, Response, render_template, redirect, url_for
from process_request import *
from helper import is_file_allowed

app = Flask(__name__)


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index page: {e}")
        return redirect(url_for('index'))


@app.route('/upload_image', methods=['POST'])
def image():
    try:
        if 'file' not in request.files:
            return redirect(url_for('index'))

        file = request.files['file']
        if file is None or file.filename == '':
            return redirect(url_for('index'))

        if is_file_allowed(file):
            return process_image(file)

        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error processing image: {e}")
        print(f"{e.__traceback__}")
        return redirect(url_for('index'))


@app.route('/upload_video', methods=['POST'])
def video():
# try:
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file is None or file.filename == '':
        return redirect(url_for('index'))

    if is_file_allowed(file):
        # return process_video(file)
        return process_video_sort(file)

    return redirect(url_for('index'))
# except Exception as e:
#     print(f"Error processing video: {e}")
#     return redirect(url_for('index'))


@app.route('/realtime')
def realtime():
    try:
        return render_template('realtime_feed.html')
    except Exception as e:
        print(f"Error rendering real-time feed page: {e}")
        return redirect(url_for('index'))


@app.route('/api/realtime_feed')
def realtime_feed():
    try:
        # return Response(generate_realtime_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
        # return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        return Response(generate_frames_sort(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error generating real-time feed: {e}")
        return redirect(url_for('index'))


@app.route('/api/realtime_results')
def realtime_results():
    return jsonify({'plates': realtime_recognized_plates})


@app.route('/history')
def history():
    pass


if __name__ == '__main__':
    app.run()
