import csv

import numpy

from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():
    """
    """

    # default params - alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01
    # It can be changed as per requirements.
    model = ElasticNetModel() 
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[v for k,v in datum.items() if k=='y'] for datum in data])

    results = model.fit(X,y)
    preds = results.predict(X)
    print(preds)
    # assert preds == 0.5