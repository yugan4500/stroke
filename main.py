from sklearn.pipeline import Pipeline
import pickle
import numpy as np


# Male,38,0,0,No,Self-employed,Urban,74.09,39.6,never smoked,0
# gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
# ever_married_dict = {'No': 0, 'Yes': 1}
# work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
# residence_type_dict = {'Rural': 0, 'Urban': 1}
# smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked':2, 'smokes': 3}


stroke = False

with open('stroke_predictor.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

def prediction(values):
    prediction = loaded_model.predict(values)
    prediction = prediction
    string = (str(prediction)[1:-1])
    return string


def test():
    if stroke == True:
        own_X = [[0, 67, 0, 1, 1, 3, 1, 228.69, 36.6, 2]]
        result = prediction(own_X)
        print(result)
    else:
        own_X = [[0, 38, 0, 0, 1, 4, 1, 74.09, 39.6, 1]]
        result = prediction(own_X)
        print(result)

def query():
    gender = int(input('gender (Male = 0, Female = 1, Unknown = 2: '))
    age = int(input('age: '))
    hypertension = int(input('hypertension (0 for false, 1 for true): '))
    heartdisease = int(input('heart_disease (0 for false 1 for true): '))
    evermarried = int(input('ever married (0 for No, 1 for yes: '))
    worktype = int(input('work type (children is 0, never worked is 1, governement job is 2, private job is 3, self employed is 4): '))
    residencetype = int(input('residence type (urban is 1 and rural is 0): '))
    avgglucoselevel = float(input('Average Glucose Level: '))
    bmi = float(input('bmi: '))
    smokingstatus = int(input('Smoking Status (0 for unknown, 1 for never smoked, 2 for formerly smoked, 3 for smokes): '))
    array = np.array([[gender, age, hypertension, heartdisease, evermarried, worktype, residencetype, avgglucoselevel, bmi, smokingstatus]])
    result = prediction(array)
    return result

def main():
    input = query()
    if int(input) == 1:
        print('Stroke is likely')
    elif int(input) == 0:
        print('Stroke is unlikely')
    else:
        print("Something went wrong")


main()