# SyntheticOversampling

This repo cotains the replication package for the paper
xxxxx (update when paper up on arxiv)

## Replication Steps
1. clone the repo with
   ```
   git clone https://github.com/XiaoLing941212/SyntheticOversampling.git
   cd SyntheticOversampling
   ```
2. create virtual environment, active virtual environment, and install required packages (example with Ubuntu 22.04)
   ```
   python -m venv [vir_name]
   source [vir_name]/bin/activate
   pip install -r requirements.txt
   ```

3. Run the experiment with following command
   ```
   python ./src/runner.py [case_study_name] [repeats] [random_projection_threshold]
   ```
   1. **case_study_name**: In the current replication package, we include 6 case studies. The name of these 6 case studies are
         1. JS_Vuln
         2. Ambari_Vuln
         3. Moodle_Vuln
         4. Defect_Eclipse_JDT_Core
         5. Defect_Eclipse_PDE_UI
         6. Defect_Mylyn
         
         Note: If you want to include your own case studies, add your data file into **/data** folder, and create the write folder in both **/result** and **/evaluation** folders. Add your own file reading script in **/src/utils.py** under the **read_data** function started from line 53.
   2. **repeats**: How many repeats you want the experiments to run? We run 10 repeats. You can choose your own from 1 to N (don't make N too large).
   3. **random_projection_threshold**: The parameter to control the size of cluster in recursive random projection algorithm.
  
4. With the results from previous steps, this step cleans the results and save them to multiple txt file for the purpose of doing statistical test
   ```
   python ./src/process_result.py [case_study_name]
   ```
   All processed txt files are saved in the **\evaluation** folder.

5. The last step is to run the statistical test. Please note that the script for statistical test is developed in Python2. Hence you need to have Python2 to run the last step.
   ```
   cat ./evaluation/[case_study_name]/[metric_name]_[learner].txt | python2 ./evaluation/sk_stats.py --text [text_size] --latex [True/False]
   ```
   1. **metric_name**: The metric to evaluate the prediction performance. The available metrics are
        1. auc
        2. recall
        3. fpr
        4. f1
        5. gscore
     
      We use recall, fpr, and gscore in our paper.
   2. **learner**: The name of the machine learning models. They are [Adaboost, DT, GBDT, KNN, LightGBM, LR, RF, SVM]
   3. **text_size**: The text size to display the statistical results in the terminal
   4. **latex**: Whether to display the latex version of generated results.
