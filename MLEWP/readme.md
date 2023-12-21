## Notes for the chapter
- [ ] 1.1 Basics
  - **conda env create -f environment.yml**
    - conda env create: This is the command to create a new Conda environment.
    - -f environment.yml: This is the path to the environment.yml file that contains the list of packages to be installed.
    Below is example of how to create a .yml file
      - name: my_environment
      - channels:
          - defaults
      - dependencies:
        - python=3.8
        - numpy=1.20
        - pandas=1.2
        - scikit-learn=0.24


### Chapter 3 -From Model to Model Factory

- training system
- model store
- Drift Detector<br>

#### Training system-> Model store -> Prediction System-> Drift Detector

#### AdaDelta: this is an extension of AdaGrad that doesn't use all the previous gradient updates. instead uses rolling window update

#### Feature vector normalization

### Training system design options

- Train run
- Train persist

#### Retraining Required

- Drift
  - Concept drift/covariate drift
  - Data drift
  - Detecting Data Drift
    - alibi-detect
    - Kolmogorov-Smirnov test
    
      - Label Drift
     
  
  - Detecting Concept Drift
  
    - online drift detection
    - Untrained AutoEncoders as processing mehtods
    - reference dataset, expected run time, window size, bootstrapped simulation to calculate threshold
  
  - Setting the limits
    - service level agreements(SLAs)
  - Diagnosing the drift
    - model dependent method  
      - mean decrease impurity(MDI) Ginni Importance
        - the feature importance have been calculated using an impurity-based feature importance measure
          that can exhibit bias towards high cardinality features and are computed only on the training data
          meaning they don't take into generalization of model onto unseen data
    - model agnostic methods
      - permutation importance
        - permutation importance is a model agnostic method that can be used to calculate feature importance
          for any black box model. It works by randomly shuffling the values of each feature and measuring the
          impact on the model's performance. The features that have the greatest impact on the model's performance
          are considered to be the most important features.
    - SHAP values
        - SHAP values are a model agnostic method that can be used to calculate feature importance for any black box model.
            SHAP values are based on Shapley values from game theory and are calculated by averaging the marginal contribution
            of each feature across all possible feature combinations. The features that have the greatest impact on the model's
            performance are considered to be the most important features.
  - Remediating the Drift
    - remove features and retrain
    - retrain with more data
    - roll back the model
    - rewrite or debug the model
    - rewrite or debug the solution
    - Fix the data source

### Automating training

      




