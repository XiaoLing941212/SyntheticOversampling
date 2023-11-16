import pandas as pd
import os
import statistics
import sys

project = sys.argv[1]
result_path = f"{os.getcwd()}/result/{project}"
result_files = [f for f in os.listdir(result_path) if f.endswith("csv")]

recall = {}
fpr = {}
AUC = {}
F_score = {}
G_score = {}

for f in result_files:
    df = pd.read_csv(f"{os.getcwd()}/result/{project}/{f}")
    
    for _, row in df.iterrows():
        learner = row['learner']
        oversampling = row['oversampling_scheme']
        
        if learner not in recall:
            recall[learner] = {}
            
        if learner not in fpr:
            fpr[learner] = {}
        
        if learner not in AUC:
            AUC[learner] = {}
            
        if learner not in F_score:
            F_score[learner] = {}
            
        if learner not in G_score:
            G_score[learner] = {}
        
        if oversampling not in recall[learner]:
            recall[learner][oversampling] = []
            
        if oversampling not in fpr[learner]:
            fpr[learner][oversampling] = []
        
        if oversampling not in AUC[learner]:
            AUC[learner][oversampling] = []
            
        if oversampling not in F_score[learner]:
            F_score[learner][oversampling] = []
            
        if oversampling not in G_score[learner]:
            G_score[learner][oversampling] = []
        
        recall[learner][oversampling].append(row['recall'])
        fpr[learner][oversampling].append(row['fpr'])
        AUC[learner][oversampling].append(row['auc'])
        F_score[learner][oversampling].append(row['f1'])
        G_score[learner][oversampling].append(row['g_score'])

for key in recall.keys():
    with open(f"{os.getcwd()}/evaluation/{project}/recall_{key}.txt", "w") as f:
        for subkey in recall[key]:
            f.write(subkey)
            f.write('\n')
            f.write("    ".join([str(item) for item in recall[key][subkey]]))
            f.write('\n')
            f.write('\n')

for key in fpr.keys():
    with open(f"{os.getcwd()}/evaluation/{project}/fpr_{key}.txt", "w") as f:
        for subkey in fpr[key]:
            f.write(subkey)
            f.write('\n')
            f.write("    ".join([str(item) for item in fpr[key][subkey]]))
            f.write('\n')
            f.write('\n')

for key in AUC.keys():
    with open(f"{os.getcwd()}/evaluation/{project}/auc_{key}.txt", "w") as f:
        for subkey in AUC[key]:
            f.write(subkey)
            f.write('\n')
            f.write("    ".join([str(item) for item in AUC[key][subkey]]))
            f.write('\n')
            f.write('\n')

for key in F_score.keys():
    with open(f"{os.getcwd()}/evaluation/{project}/f1_{key}.txt", "w") as f:
        for subkey in F_score[key]:
            f.write(subkey)
            f.write('\n')
            f.write("    ".join([str(item) for item in F_score[key][subkey]]))
            f.write('\n')
            f.write('\n')

for key in G_score.keys():
    with open(f"{os.getcwd()}/evaluation/{project}/gscore_{key}.txt", "w") as f:
        for subkey in G_score[key]:
            f.write(subkey)
            f.write('\n')
            f.write("    ".join([str(item) for item in G_score[key][subkey]]))
            f.write('\n')
            f.write('\n')

runtime = {}
for f in result_files:
    df = pd.read_csv(f"{os.getcwd()}/result/{project}/{f}")
    for _, row in df.iterrows():
        oversampling = row['oversampling_scheme']
        
        if oversampling not in runtime:
            runtime[oversampling] = []
        
        runtime[oversampling].append(row['runtime'])

for key in runtime.keys():
    runtime[key] = statistics.mean(runtime[key])

print(runtime)