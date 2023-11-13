from howso import engine
from howso.engine import Trainee
from howso.utilities import infer_feature_attributes
import time

def howsoOversampling(X_train, y_train):
    start_time = time.time()
    tar = y_train.name
    X_train[tar] = y_train

    features = infer_feature_attributes(X_train)
    for f_name, f_value in features.items():
        if f_value["type"] == "nominal":
            f_value["non_sensitive"] = True

    context_shape = int(X_train.shape[0] / 2)
    contexts = [[1]] * context_shape + [[0]] * context_shape
    context_features = [tar]

    t = Trainee(
        features=features,
        overwrite_existing=True
    )

    t.train(X_train)
    t.analyze()

    synth = t.react(desired_conviction=5,
                    generate_new_cases="no",
                    contexts=contexts,
                    context_features=context_features,
                    num_cases_to_generate=len(contexts))

    rt = time.time() - start_time
    X_new = synth['action']
    print(X_new)
    X_train_new = synth['action'].iloc[:, :-1]
    y_train_new = synth['action'].iloc[:, -1]

    return rt, X_train_new, y_train_new

import os
from scipy.io import arff
import pandas as pd
import random
from sklearn.model_selection import train_test_split
data_path = f"{os.path.dirname(os.getcwd())}\\SyntheticOversampling\\data\\Vulnerable_Files\\moodle-2_0_0-metrics.arff"
data = arff.loadarff(data_path)
df = pd.DataFrame(data[0])
df['IsVulnerable'] = df['IsVulnerable'].astype('str')
d = {'b\'yes\'': 1, 'b\'no\'': 0}
df['IsVulnerable'] = df['IsVulnerable'].astype(str).map(d).fillna(df['IsVulnerable'])
df = df.drop_duplicates()
df.reset_index(inplace=True, drop=True)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
rs = random.randint(0, 100000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
rt, X_train_new, y_train_new = howsoOversampling(X_train=X_train, y_train=y_train)
print(X_train_new)
print(y_train_new)
print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))
