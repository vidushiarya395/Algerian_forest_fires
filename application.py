# algerian forest fires predicton project
from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

#import scaler and ridgereg
ridge_model= pickle.load(open('models/ridge.pkl','rb'))
standard_scaler= pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/predictdata' ,methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data)
        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)









    """
====================================================================
Git Workflow Reference - Flask Project
====================================================================

1️⃣ Local Repository:
- The folder 'flask_lab_template' with the '.git/' folder is your local repo.
- All changes you make (edits, comments, new files) exist here first.
- Changes are NOT tracked by Git until you stage and commit them.

2️⃣ Remote Repository:
- This is your online repo on GitHub:
  https://github.com/vidushiarya395/flask_lab_template.git
- Used for backup, sharing, and collaboration.
- Push local commits to the remote using 'git push'.
- Fetch or pull updates from remote using 'git fetch' or 'git pull'.


4️⃣ Typical Git Workflow:
- Make changes in local files (add comments, edit code, etc.)
- Stage changes:
    git add <file>      # or git add . for all changes
- Commit changes to local repo:
    git commit -m "Describe your changes"
- Push commits to remote repo:
    git push

5️⃣ Notes:
- Until you commit, changes are only in your **working copy** (local files).
- After commit, changes are tracked in the **local repository**.
- After push, changes are on the **remote repository (GitHub)**.
- Use 'git remote -v' to check fetch/push links.

====================================================================
"""
