from flask import Flask, request, redirect, url_for, render_template, flash, session
from werkzeug.utils import secure_filename
import os
import PIL
from PIL import Image
from classifier import Classifier_library
import random
import datetime

BASE_PATH = os.path.abspath('.')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
THUMB_WIDTH = 640
DEFAULT_PREDICT_BACKGROUND = 'streets.jpg'
EXAMPLES = ['shopping.jpg', 'plane.jpg', 'driving4.jpg', 'public2.jpg', 'meeting.jpg', 'biking2.jpg', 'cleaning.jpg']
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/hermetico/dev/flask-activity-classifier/static/upload/'
#app.secret_key = str(datetime.datetime.utcnow()) + ' such  a secret!'
app.secret_key = ' such  a secret!'
classifier_lib = Classifier_library()


"""
@app.route('/test')
def test():
    image_path = '/home/hermetico/dev/annotation-tool/annotation-tool/app/static/media/1/2015-05-05/b00005118_21i7lf_20150505_095946e.jpg'
    label = classifier_lib.classify(image_path)
    return 'The label is: ' + label
"""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_thumbnail(name):
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thumb', name)

    img = Image.open(path)
    wpercent = (THUMB_WIDTH / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((THUMB_WIDTH, hsize), PIL.Image.ANTIALIAS)
    img.save(new_path)

@app.route('/')
@app.route('/classify/<path>')
def main(path=None):
    cp_examples = EXAMPLES[:]
    examples = []
    if not 'previous-pics' in session:
        session['previous-pics'] = {}

    for x in range(4):
        num = random.randint(0, len(cp_examples)-1)
        examples.append(cp_examples[num])
        cp_examples.pop(num)

    examples = random.sample(EXAMPLES, 4)
    if path is not None:
        path = path.replace(os.path.pathsep, '')  # security reasons
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], path)
        print "classifying " + path
        label = classifier_lib.classify(filepath)
        top5_labels, top5_percent = classifier_lib.classify(filepath, way='cnn')
        top5 = zip(top5_labels, top5_percent)
        if path not in examples:  # if not, it is a test picture
            session['previous-pics'][path] = label
        return render_template('index.html', data=dict(label=label, top5=top5, classifier_background=path), examples=examples, previous_pics=session['previous-pics'])

    return render_template('base.html', examples=examples, previous_pics=session['previous-pics'])

@app.route('/upload', methods=['POST'])
def upload():
    if not 'previous-pics' in session:
        session['previous-pics'] = []

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print "not file object"
            flash('No file part')
            return redirect(url_for('.main', _anchor='classify'))
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print "no file selected"
            flash('No selected file')
            return redirect(url_for('.main', _anchor='classify'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # modify filenames
            while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                filename = str(random.randint(0, 9)) + filename

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            save_thumbnail(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            session['previous-pics'][filename] = ''

            print "path: " + filepath
            return redirect(url_for('.main', path=filename, _anchor='prediction'))
        else:
            print "Filetype not allowed"
            flash("Filetype not allowed")
            return redirect(url_for('.main', _anchor='classify'))
    else:
        return redirect(url_for('.main'))

"""
@app.route('/testindex')
def test_index():
    return render_template('test_index.html')
"""

if __name__ == '__main__':
    app.run(debug=True)
