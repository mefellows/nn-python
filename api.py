import numpy
from flask import Flask, request, abort, jsonify
from net import NeuralNetwork

# Network parameters
input_nodes = 20
hidden_nodes = 100
output_nodes = 1
learning_rate = 0.2 # Make this .02 if you reduce epochs
epochs = 10
score_threshold = 0.09 # Anything above this is considered 'yes', customer is likely to subscribe

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

from numpy import genfromtxt
training_data_list = genfromtxt('./banking-train.csv', delimiter=',', skip_header=1, dtype=str)

# Map well known strings to numbers
job_categories = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
marital_categories = ['divorced', 'married', 'single', 'unknown']
education_categories = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
default_categories = ['no', 'unknown', 'yes']
housing_categories = ['no', 'unknown', 'yes']
loan_categories = ['no', 'unknown', 'yes']
contact_categories = ['cellular', 'telephone']
month_categories = ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep']
day_of_week_categories = ['fri', 'mon', 'thu', 'tue', 'wed']
poutcome_categories = ['failure', 'nonexistent', 'success']

# Make each column a record so that we can normalise them easily
rotated_values = numpy.asarray(training_data_list).T

# age,job,marital,education,default,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,emp_var_rate,cons_price_idx,cons_conf_idx,euribor3m,nr_employed,y
# Map the column index of the data to a mapping array.
# If not in array, normalise the data
mappings = {
    1: job_categories,
    2: marital_categories,
    3: education_categories,
    4: default_categories,
    5: housing_categories,
    6: loan_categories,
    7: contact_categories,
    8: month_categories,
    9: day_of_week_categories,
    14: poutcome_categories
}

# For all of the floating points,
normalised_data = {}
for i in [0,10, 11, 12, 13, 15, 16, 17, 18, 19]:
    col = numpy.asfarray(rotated_values[i])
    minx = min(col)
    maxx = max(col)
    normalised_data[i] = {"col": col, "minx": minx, "maxx": maxx}

def prepare_inputs(input, process_target=True):
    """Given a CSV matrix convert it to inputs and target outputs."""
    inputs_list = []
    targets_list = []
    for record in input:
        inputs = []
        targets = []
        data = len(record) - 1 if process_target else len(record)
        for i in range(data):
            if i in mappings:
                inputs.append(mappings.get(i).index(record[i]) + 0.01)
            else:
                xi = float(record[i])
                col = normalised_data[i]["col"]
                minx = normalised_data[i]["minx"]
                maxx = normalised_data[i]["maxx"]
                normalised = (xi - minx) / (maxx - minx)
                inputs.append(normalised)

        inputs_list.append(inputs)

        if (process_target):
            targets = numpy.zeros(output_nodes) + 0.01
            targets[0] = 0.99 if int(record[-1:][0]) > 0 else 0.01
            targets_list.append(targets)

    return (inputs_list, targets_list)

def score_output(output):
    value = output.flatten().max()
    if value > score_threshold:
        return 1

    return 0

# Main training loop:
# - Format training data
# - Train network on said data
print("Training model...")
for e in range(epochs):
    inputs, targets = prepare_inputs(training_data_list)
    for i, input in enumerate(inputs):
        n.train(inputs[i], targets[i])

print("Training complete!")
print("")
print("Testing model fit...")
# Import test data
test_data_list = genfromtxt('./banking-test.csv', delimiter=',', skip_header=1, dtype=str)

scorecard = []
yes_actual = 0
no_actual = 0
inputs, targets = prepare_inputs(test_data_list)
yes_target = len(list(filter(lambda v: float(v) > 0.5, targets)))
no_target = len(test_data_list) - yes_target

for i, record in enumerate(inputs):
    correct_label = 1 if float(targets[i]) > 0.5 else 0
    outputs = n.query(inputs[i])
    value = outputs.flatten().max()
    label = 0

    if value > score_threshold:
        label = 1
        yes_actual += 1

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

no_actual = len(test_data_list) - yes_actual
scorecard_array = numpy.asarray(scorecard)
print("Scorecard: {}%".format(scorecard_array.sum() / scorecard_array.size * 100))
print(f"Target number of yes: {yes_target}, Target number of no: {no_target}")
print(f"Actual yes: {yes_actual}, Actual number of no: {no_actual}")

# Individual
inputs, _ = prepare_inputs([["32","services","divorced","basic.9y","no","unknown","yes","cellular","dec","mon","110","1","11","0","nonexistent","-1.8","94.465","-36.1","0.883","5228.1"]], False)
output = score_output(n.query(inputs))
print(f"Individual (real-time) score: {output}")

# API
app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    if not request.json or not 'input' in request.json:
        abort(400)

    print(request.json['input'])
    inputs, _ = prepare_inputs([request.json['input']], False)
    output = score_output(n.query(inputs))
    print(f"Individual (real-time) score: {output}")
    return jsonify({ "response": output })

@app.route('/health', methods=['GET'])
def health():
    return 'up'

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
