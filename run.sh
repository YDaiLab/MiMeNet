python MiMeNet_train.py -micro data/IBD/microbiome_PRISM.csv -metab data/IBD/metabolome_PRISM.csv -external_micro data/IBD/microbiome_external.csv -external_metab data/IBD/metabolome_external.csv -micro_norm None -metab_norm CLR -net_params results/IBD/network_parameters.txt -annotation data/IBD/metabolome_annotation.csv -labels data/IBD/diagnosis_PRISM.csv -num_run_cv 10 -num_background 20 -output IBD

python MiMeNet_train.py -micro data/cystic_fibrosis/microbiome.csv -metab data/cystic_fibrosis/metabolome.csv  -micro_norm CLR -metab_norm CLR  -num_run_cv 10 -output cystic_fibrosis -net_params results/cystic_fibrosis/network_parameters.txt -num_background 20

python MiMeNet_train.py -micro data/soil/microbiome.csv -metab data/soil/metabolome.csv  -micro_norm CLR -metab_norm CLR  -num_run_cv 10 -output soil -net_params results/soil/network_parameters.txt -num_background 20


