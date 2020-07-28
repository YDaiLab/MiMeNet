## MiMeNet: Exploring Microbiome-Metabolome Relationships using Neural Networks
MiMeNet predicts the metabolomic profile from microbiome data and learns undelrying relationships between the two.

### Prerequisites
* MiMeNet is tested to work on Python 3.7+
* MiMeNet requires the following Python libraries:
  - Tensorflow 1.14
  - Numpy 1.17.2
  - Pandas 0.25.1
  - Scipy 1.3.1
  - Scikit-learn 0.21.3
  - Matplotlib 3.0.3
  - Seaborn 0.9.0

A Conda Python environment is provided in pseudocell_tracer.yml


### Usage

```bash
usage: MiMeNet_train.py [-h] -micro MICRO -metab METAB
                        [-external_micro EXTERNAL_MICRO]
                        [-external_metab EXTERNAL_METAB]
                        [-annotation ANNOTATION] [-labels LABELS] -output
                        OUTPUT [-net_params NET_PARAMS]
                        [-background BACKGROUND]
                        [-num_background NUM_BACKGROUND]
                        [-micro_norm MICRO_NORM] [-metab_norm METAB_NORM]
                        [-threshold THRESHOLD] [-num_run_cv NUM_RUN_CV]
                        [-num_cv NUM_CV] [-num_run NUM_RUN]

```

```
  -h, --help                        Show this help message and exit
  -micro MICRO                      Comma delimited file representing matrix of samples by microbial features
  -metab METAB                      Comma delimited file representing matrix of samples by metabolomic features
  -external_micro EXTERNAL_MICRO    Comma delimited file representing matrix of samples by microbial features
  -external_metab EXTERNAL_METAB    Comma delimited file representing matrix of samples by metabolomic features
  -annotation ANNOTATION            Comma delimited file annotating subset of metabolite features
  -labels LABELS                    Comma delimited file for sample labels to associate clusters with
  -output OUTPUT                    Output directory
  -net_params NET_PARAMS            JSON file of network hyperparameters
  -background BACKGROUND            Directory with previously generated background
  -num_background NUM_BACKGROUND    Number of background CV Iterations
  -micro_norm MICRO_NORM            Microbiome normalization (RA, CLR, or None)
  -metab_norm METAB_NORM            Metabolome normalization (RA, CLR, or None)
  -threshold THRESHOLD              Define significant correlation threshold
  -num_run_cv NUM_RUN_CV            Number of iterations for cross-validation
  -num_cv NUM_CV                    Number of cross-validated folds
  -num_run NUM_RUN                  Number of iterations for training full model
```

### Example for provided dataset

```bash
python MiMeNet_train.py -micro data/IBD/microbiome_PRISM.csv -metab data/IBD/metabolome_PRISM.csv \
                        -external_micro data/IBD/microbiome_external.csv -external_metab data/IBD/metabolome_external.csv \
                        -micro_norm None -metab_norm CLR -net_params results/IBD/network_parameters.txt \
                        -annotation data/IBD/metabolome_annotation.csv -labels data/IBD/diagnosis_PRISM.csv \
                        -num_run_cv 10 -output IBD
```

The provided command will run MiMeNet on the IBD dataset and store results in the directory _results/output_dir_. 


### Version
1.0.0 (2020/07/28)

### Publication
TBA

### Contact
* Please contact Derek Reiman <dreima2@uic.edu> or Yand Dai <yangdai@uic.edu> for any questions or comments.

### License
Software provided to academic users under MIT License
