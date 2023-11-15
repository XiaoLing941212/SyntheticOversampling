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
   python .\src\runner.py [case_study_name] [repeats] [random_projection_threshold]
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
  
4. 

## Algorithms and References
1. **Algorithm**: SMOTE and its variants    
   **Reference**: Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning
