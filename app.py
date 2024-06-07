from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
app.secret_key = '12345'

# Example training data with 31 features (replace with actual training data)
X_train = np.random.rand(100, 31)  # 100 samples, 31 features
y_train = np.random.randint(0, 2, 100)  # Binary target variable: 0 or 1

# Initialize and fit the scaler and model with the training data
scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # For simplicity, accepting any email and password
        session['email'] = email
        
        # Redirect to the diagnosis page only if the login is successful
        return redirect(url_for('diagnosis'))
        
    return render_template('login.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        features_str = request.form['features']
        features_list = [float(x) for x in features_str.split(',')]
        
        if len(features_list) != 31:
            return render_template('diagnosis.html', prediction="Error: Expected 31 features")
        
        features_array = np.array([features_list])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction_numeric = model.predict(features_scaled)[0]
        
        # Map prediction to 'M' or 'B'
        prediction = 'Malignant' if prediction_numeric == 1 else 'Benign'
        
        return render_template('diagnosis.html', prediction=prediction)
    
    return render_template('diagnosis.html', prediction=None)

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
