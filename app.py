from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open(r'C:\Users\Mekonen\Documents\New folder (10)\Crop-Recommendation-System-Using-Machine-Learning-main\model\model.pkl', 'rb'))
sc = pickle.load(open(r'C:\Users\Mekonen\Documents\New folder (10)\Crop-Recommendation-System-Using-Machine-Learning-main\model\standscaler.pkl','rb'))
ms = pickle.load(open(r'C:\Users\Mekonen\Documents\New folder (10)\Crop-Recommendation-System-Using-Machine-Learning-main\model\minmaxscaler.pkl','rb')) 

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1).astype(float)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Get top 3 predictions with probabilities
    probabilities = model.predict_proba(final_features)[0]
    top3_indices = np.argsort(probabilities)[::-1][:3]

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    top3 = [(crop_dict[i+1], round(probabilities[i]*100, 2)) for i in top3_indices]

    result = "Top 3 recommended crops:<br>"
    for rank, (crop, prob) in enumerate(top3, start=1):
        result += f"{rank}. {crop} ({prob}%)<br>"

    return render_template('index.html', result=result)





# python main
if __name__ == "__main__":
    app.run(debug=True)