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
