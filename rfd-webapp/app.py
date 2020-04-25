# Importing libraries
import praw
import pandas as pd
import tldextract
import cleaning
import pickle
import flask
from flask import Flask, render_template, request, make_response, jsonify
import requests
from werkzeug import secure_filename

# Loading pickled vectorizer, LSA topics matrix and ML model
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("LSA_topics.pkl", "rb") as file:
    tsvd = pickle.load(file)

with open("svm_model.pkl", "rb") as file:
    classifier = pickle.load(file)

# Authenticating reddit usage
reddit = praw.Reddit(client_id='SPqlEvQipFXnVg', client_secret='Wo_MBHLnSx5RPD9okevEoHlt6xA', user_agent='Harshita Chopra')

flairs = ['AskIndia', 'Business/Finance', 'Food', 'Non-Political', 'Photography', 'Policy/Economy', 'Politics', 'Scheduled',
          'Science/Technology', 'Sports']

# Function to predict flair by accepting URL
def predict_flair(url):
    submission = reddit.submission(url=url)
    flair = submission.link_flair_text
    title = submission.title
    body = submission.selftext

    # Using top-level-domain extraction methods to find domain of URLs
    tld = tldextract.extract(submission.url)
    domain = tld.domain + "." + tld.suffix

    # Conditions for some exceptions
    if submission.is_self == True:
        domain = "self-post"
    if domain == "youtu.be":
        domain = "youtube.com"
    if domain == "redd.it":
        domain = "reddit.com"

    # Extracting top comments
    submission.comments.replace_more(limit=10)
    comments = ''
    for top_level_comment in submission.comments:
        comments += ' ' + top_level_comment.body

    content = title +' '+ body +' '+ comments +' '+ domain
    data = pd.DataFrame({"content":[content]})

    # Cleaning the content and making prediction
    cleaning.clean_text(data, "content")
    X = tfidf.transform(data.content).toarray()
    X = tsvd.transform(X)
    y_pred = classifier.predict(X)

    return flairs[int(y_pred)], title

# CREATING THE FLASK APP

app = Flask(__name__, template_folder='templates')

ALLOWED_EXTENSIONS = ['txt']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        post_url = flask.request.form['url']
        prediction, post_title = predict_flair(post_url)

        return flask.render_template('index.html',
                                     flair = prediction, title = f'Title - {post_title}'
                                    )

@app.route('/automated_testing', methods=['GET','POST'])
def getfile():
    if request.method == 'POST':
        for file in request.files:
            required_file = request.files[file]
            if required_file and allowed_file(required_file.filename):
                print("Text file found")
                filename = secure_filename(required_file.filename)
                required_file.save(filename)

                with open(filename,'r') as f:
                    file_content = f.read().splitlines()

                pred = {}
                for url in file_content:
                    pred[url], _ = predict_flair(url)

                return make_response(jsonify(pred))

            return 'Invalid File'
    else:
        return '''
    <!doctype html>
    <title>Automated testing</title>
    <h3>Send a POST request to this webpage with a text file containing URLs, one in each line. <br><br> The response will be a json file containing keys as URLs and values as predicted flair. </h3>
    '''

# run the application
if __name__ == "__main__":
    app.run()
