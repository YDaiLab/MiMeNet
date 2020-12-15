import warnings
warnings.filterwarnings("ignore")
import os

import biom
import json
import argparse
import time
import datetime
import json
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.stats import spearmanr, mannwhitneyu
import scipy.cluster.hierarchy as shc
from skbio.stats.composition import clr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import cut_tree

from src.models.MiMeNet import MiMeNet, tune_MiMeNet


###################################################
# Read in command line arguments
###################################################

parser = argparse.ArgumentParser(description='Perform MiMeNet')

parser.add_argument('-micro', '--micro', help='Comma delimited file representing matrix of samples by microbial features', required=True)
parser.add_argument('-metab', '--metab', help= 'Comma delimited file representing matrix of samples by metabolomic features', required=True)
parser.add_argument('-external_micro', '--external_micro', help='Comma delimited file representing matrix of samples by microbial features')
parser.add_argument('-external_metab', '--external_metab', help= 'Comma delimited file representing matrix of samples by metabolomic features')
parser.add_argument('-annotation', '--annotation', help='Comma delimited file annotating subset of metabolite features')
parser.add_argument('-labels', '--labels', help="Comma delimited file for sample labels to associate clusters with")
parser.add_argument('-output', '--output', help='Output directory', required=True)

parser.add_argument('-net_params', '--net_params', help='JSON file of network hyperparameters', default=None)
parser.add_argument('-background', '--background', help='Directory with previously generated background', default=None)
parser.add_argument('-num_background', '--num_background', help='Number of background CV Iterations', default=100, type=int)

parser.add_argument('-micro_norm', '--micro_norm', help='Microbiome normalization (RA, CLR, or None)', default='CLR')
parser.add_argument('-metab_norm', '--metab_norm', help='Metabolome normalization (RA, CLR, or None)', default='CLR')
parser.add_argument('-threshold', '--threshold', help='Define significant correlation threshold', default=None)
parser.add_argument('-num_run_cv', '--num_run_cv', help='Number of iterations for cross-validation', default=1, type=int)
parser.add_argument('-num_cv', '--num_cv', help='Number of cross-validated folds', default=10, type=int)
parser.add_argument('-num_run', '--num_run', help='Number of iterations for training full model', type=int, default=10)

args = parser.parse_args()

micro = args.micro
metab = args.metab
external_micro = args.external_micro
external_metab = args.external_metab
annotation = args.annotation
out = args.output
net_params = args.net_params
threshold = args.threshold
micro_norm = args.micro_norm
metab_norm = args.metab_norm
num_run_cv = args.num_run_cv
num_cv = args.num_cv
num_run = args.num_run
background_dir = args.background
labels = args.labels
num_bg = args.num_background

tuned = False

gen_background = True
if background_dir != None:
    gen_background = False


start_time = time.time()

if external_metab != None and external_micro == None:
    print("Warning: External metabolites found with no external microbiome...ignoring external set!")
    external_metab = None

if net_params != None:
    print("Loading network parameters...")
    try:
        with open(net_params, "r") as infile:
            params = json.load(infile)
            num_layer = params["num_layer"]
            layer_nodes = params["layer_nodes"]
            l1 = params["l1"]
            l2 = params["l2"]
            dropout = params["dropout"]
            learning_rate = params["lr"]
            tuned = True
            print("Loaded network parameters...")
    except:
        print("Warning: Could not load network parameter file!")

###################################################
# Load Data
###################################################

metab_df = pd.read_csv(metab, index_col=0)
micro_df = pd.read_csv(micro, index_col=0)


if external_metab != None:
    external_metab_df = pd.read_csv(external_metab, index_col=0)

if external_micro != None:
    external_micro_df = pd.read_csv(external_micro, index_col=0)
      
    
###################################################
# Filter only paired samples
###################################################

samples = np.intersect1d(metab_df.columns.values, micro_df.columns.values)
num_samples = len(samples)

metab_df = metab_df[samples]
micro_df = micro_df[samples]

for c in micro_df.columns:
    micro_df[c] = pd.to_numeric(micro_df[c])
    
for c in metab_df.columns:
    metab_df[c] = pd.to_numeric(metab_df[c])

    
if external_metab != None and external_micro != None:
    external_samples = np.intersect1d(external_metab_df.columns.values, external_micro_df.columns.values)
    external_metab_df = external_metab_df[external_samples]
    external_micro_df = external_micro_df[external_samples]

    for c in external_micro_df.columns:
        external_micro_df[c] = pd.to_numeric(external_micro_df[c])

    for c in external_metab_df.columns:
        external_metab_df[c] = pd.to_numeric(external_metab_df[c])
        
    num_external_samples = len(external_samples)

elif external_micro != None:
    external_samples = external_micro_df.columns.values
    external_micro_df = external_micro_df[external_samples]

    for c in external_micro_df.columns:
        external_micro_df[c] = pd.to_numeric(external_micro_df[c])

    num_external_samples = len(external_samples)



###################################################
# Create output directory
###################################################
    
dirName = 'results'
 
try:
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
    
dirName = 'results/' + out
 
try:
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists") 
    
dirName = 'results/' + out + "/Images"
 
try:
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists") 
    
        
    

###################################################
# Filter lowly abundant samples
###################################################

to_drop = []

for microbe in micro_df.index.values:
    present_in = sum(micro_df.loc[microbe] > 0.0000)
    if present_in <= 0.1 * num_samples:
        to_drop.append(microbe)

micro_df = micro_df.drop(to_drop, axis=0)

to_drop = []

for metabolite in metab_df.index.values:
    present_in = sum(metab_df.loc[metabolite] > 0.0000)
    if present_in <= 0.1 * num_samples:
        to_drop.append(metabolite)

metab_df = metab_df.drop(to_drop, axis=0)

if external_micro != None:
    common_features = np.intersect1d(micro_df.index.values, external_micro_df.index.values)
    micro_df = micro_df.loc[common_features]
    external_micro_df = external_micro_df.loc[common_features]

if external_metab != None:
    common_features = np.intersect1d(metab_df.index.values, external_metab_df.index.values)
    metab_df = metab_df.loc[common_features]
    external_metab_df = external_metab_df.loc[common_features]
    

    

    
###################################################
# Transform data to Compositional Data
###################################################

# Transform Microbiome Data
if micro_norm == "CLR":
    micro_comp_df = pd.DataFrame(data=np.transpose(clr(micro_df.transpose() + 1)), 
                                     index=micro_df.index, columns=micro_df.columns)
    if external_micro:
        external_micro_comp_df = pd.DataFrame(data=np.transpose(clr(external_micro_df.transpose() + 1)), 
                                     index=external_micro_df.index, columns=external_micro_df.columns)
elif micro_norm == "RA":
    col_sums = micro_df.sum(axis=0)
    micro_comp_df = micro_df/col_sums  
    
    if external_micro:
        col_sums = external_micro_df.sum(axis=0)
        external_micro_comp_df = external_micro_df/col_sums
        
else:
    micro_comp_df = micro_df
    if external_micro:
        external_micro_comp_df = external_micro_df
        
        
        
# Normalize Metabolome Data
if metab_norm == "CLR":
    metab_comp_df = pd.DataFrame(data=np.transpose(clr(metab_df.transpose() + 1)), 
                                     index=metab_df.index, columns=metab_df.columns)
    if external_metab:
        external_metab_comp_df = pd.DataFrame(data=np.transpose(clr(external_metab_df.transpose() + 1)), 
                                     index=external_metab_df.index, columns=external_metab_df.columns)
elif metab_norm == "RA":
    col_sums = metab_df.sum(axis=0)
    metab_comp_df = metab_df/col_sums
    
    if external_metab:
        col_sums = external_metab_df.sum(axis=0)
        external_metab_comp_df = external_metab_df/col_sums
            
else:
    metab_comp_df = metab_df
    if external_metab:
        external_metab_comp_df = external_metab_df
    
    
micro_comp_df = micro_comp_df.transpose()
metab_comp_df = metab_comp_df.transpose()

if external_micro:
    external_micro_comp_df = external_micro_comp_df.transpose()

if external_metab:
    external_metab_comp_df = external_metab_comp_df.transpose()
    
    
    
###################################################
# Run Cross-Validation on Dataset
###################################################
score_matrices = []
    
print("Performing %d runs of %d-fold cross-validation" % (num_run_cv, num_cv))
cv_start_time = time.time()
tune_run_time = 0

micro = micro_comp_df.values
metab = metab_comp_df.values

dirName = 'results/' + out + '/CV' 
try:
    os.mkdir(dirName)
except FileExistsError:
    pass

for run in range(0,num_run_cv):

    # Set up output directory for CV runs
    dirName = 'results/' + out + '/CV/' + str(run) 
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass

    # Set up CV partitions
    kfold = KFold(n_splits=num_cv, shuffle=True)
    cv = 0
        
    for train_index, test_index in kfold.split(samples):

        # Set up output directory for CV partition run
        dirName = 'results/' + out + '/CV/' + str(run) + '/' + str(cv)
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass
                
        # Partition data into training and test sets
        train_micro, test_micro = micro[train_index], micro[test_index]
        train_metab, test_metab = metab[train_index], metab[test_index]
        train_samples, test_samples = samples[train_index], samples[test_index]
                
        # Store training and test set partitioning
        train_microbe_df = pd.DataFrame(data=train_micro, index=train_samples, columns=micro_comp_df.columns)
        test_microbe_df = pd.DataFrame(data=test_micro, index=test_samples, columns=micro_comp_df.columns)
        train_metab_df = pd.DataFrame(data=train_metab, index=train_samples, columns=metab_comp_df.columns)
        test_metab_df = pd.DataFrame(data=test_metab, index=test_samples, columns=metab_comp_df.columns)
                
        train_microbe_df.to_csv(dirName + "/train_microbes.csv")
        test_microbe_df.to_csv(dirName + "/test_microbes.csv")
        train_metab_df.to_csv(dirName + "/train_metabolites.csv")
        test_metab_df.to_csv(dirName + "/test_metabolites.csv")
                
        # Log transform data if RA
        if micro_norm == "RA" or micro_norm == None:
            train_micro = np.log(train_micro + 1)
            test_micro = np.log(test_micro + 1)
                
        if metab_norm == "RA" or metab_norm == None:
            train_metab = np.log(train_metab + 1)
            test_metab = np.log(test_metab + 1)
            
        # Scale data before neural network training
        micro_scaler = StandardScaler().fit(train_micro)
        train_micro = micro_scaler.transform(train_micro)
        test_micro = micro_scaler.transform(test_micro)       
                
        metab_scaler = StandardScaler().fit(train_metab)
        train_metab = metab_scaler.transform(train_metab)
        test_metab = metab_scaler.transform(test_metab)            

        # Aggregate paired microbiome and metabolomic data
        train = (train_micro, train_metab)
        test = (test_micro, test_metab)

        # Tune hyperparameters if first partition
        if tuned == False:
            tune_start_time = time.time()
            print("Tuning parameters...")
            tuned = True
            params = tune_MiMeNet(train)
            l1 = params['l1']
            l2 = params['l2']
            num_layer=params['num_layer']
            layer_nodes=params['layer_nodes']
            dropout=params['dropout']
            with open('results/' +out + '/network_parameters.txt', 'w') as outfile:
                json.dump(params, outfile)
                        
            tune_run_time = time.time() - tune_start_time
            print("Tuning run time: " + (str(datetime.timedelta(seconds=(tune_run_time)))))
                    
        print("Run: %02d\t\tFold: %02d" % (run + 1, cv + 1), end="\r")
                
        # Construct Neural Network Model
        model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=l1, l2=l2, 
                            num_layer=num_layer, layer_nodes=layer_nodes, dropout=dropout)

        #Train Neural Network Model
        model.train(train)
                
        # Predict on test set
        p = model.test(test)

        inv_p = metab_scaler.inverse_transform(p)
                
        if metab_norm == "RA" or metab_norm == None:
            inv_p = np.exp(inv_p) - 1
            inv_p = inv_p/np.sum(inv_p)
            
        score_matrices.append(model.get_scores())
        prediction_df = pd.DataFrame(data=inv_p, index=test_samples, columns=metab_comp_df.columns)
        score_matrix_df = pd.DataFrame(data=model.get_scores(), index=micro_comp_df.columns, columns=metab_comp_df.columns)

        prediction_df.to_csv(dirName + "/prediction.csv")
        score_matrix_df.to_csv(dirName + "/score_matrix.csv")

        model.destroy()
        tf.keras.backend.clear_session()

        cv += 1
            
print("\nCV run time: " + str(datetime.timedelta(seconds=(time.time() - cv_start_time - tune_run_time))))         
print("\nCalculating correlations for cross-validated evaluation...")
    
    
    
###################################################
# Calculate correlation across CV
###################################################    
        
correlation_cv_df = pd.DataFrame(index=metab_comp_df.columns)

for run in range(num_run_cv):
    preds = pd.concat([pd.read_csv('results/' + out + '/CV/' + str(run)+ '/' + str(cv) + "/prediction.csv", 
                                    index_col=0) for cv in range(0, num_cv)])
    y = pd.concat([pd.read_csv('results/' + out + '/CV/' + str(run)+ '/' + str(cv) + "/test_metabolites.csv", 
                                index_col=0) for cv in range(0, num_cv)])

    cor = y.corrwith(preds, method="spearman")
    correlation_cv_df["Run_"+str(run)] = cor.loc[correlation_cv_df.index]

correlation_cv_df["Mean"] = correlation_cv_df.mean(axis=1)
correlation_cv_df = correlation_cv_df.sort_values("Mean", ascending=False)
correlation_cv_df.to_csv('results/' + out + '/cv_correlations.csv')

fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(111)
sns.distplot(correlation_cv_df["Mean"])
plt.title("IBD Prediction Correlation")
plt.ylabel("Frequency")
plt.xlabel("Spearman Correlation")
plt.text(0.1, 0.9,"Mean: %.3f"% np.mean(correlation_cv_df.values),
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
plt.savefig('results/' + out + '/Images/cv_correlation_distribution.png')
print("Mean correlation: %f" % np.mean(correlation_cv_df.values))

    
###################################################
# Generate Background Distributions
###################################################    

if gen_background == False:
    try:
        print("Loading background from directory...")
        infile = open(background_dir + "/bg_preds.pkl", "rb")
        bg_preds = pickle.load(infile)
        infile.close()

        infile = open(background_dir + "/bg_truth.pkl", "rb")
        bg_truth = pickle.load(infile)
        infile.close()

        infile = open(background_dir + "/bg_scores_mean.pkl", "rb")
        bg_scores_mean = pickle.load(infile)
        infile.close()
        
        infile = open(background_dir + "/bg_correlations.pkl", "rb")
        bg_corr = pickle.load(infile)
        infile.close()
    except:
        print("Warning: Failed to load background from directory...")
        gen_background = True
        
if gen_background == True:
    print("Generating background using 100 10-fold cross-validated runs of shuffled data...")
    bg_preds = []
    bg_truth = []
    bg_scores = []
    bg_start_time = time.time()
    for run in range(0,num_bg):
        preds = []
        truth = []
        score_matrix = []

        micro = micro_comp_df.values
        metab = metab_comp_df.values

        np.random.shuffle(micro)
        np.random.shuffle(metab)

        kfold = KFold(n_splits=10)
        cv=0
        for train_index, test_index in kfold.split(micro):
            print("Run: %02d\t\tFold:%02d" % (run + 1, cv + 1), end="\r")
            train_micro, test_micro  = micro[train_index], micro[test_index]
            train_metab, test_metab = metab[train_index], metab[test_index]

            # Scale data before neural network training
            micro_scaler = StandardScaler().fit(train_micro)
            train_micro = micro_scaler.transform(train_micro)
            test_micro = micro_scaler.transform(test_micro)       

            metab_scaler = StandardScaler().fit(train_metab)
            train_metab = metab_scaler.transform(train_metab)
            test_metab = metab_scaler.transform(test_metab)        

            train = (train_micro, train_metab)
            test = (test_micro, test_metab)

            model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=l1, l2=l2, 
                            num_layer=num_layer, layer_nodes=layer_nodes, dropout=dropout)

            model.train(train)
            p = model.test(test)

            preds = list(preds) + list(p)
            truth = list(truth) + list(test_metab)
            score_matrix.append(model.get_scores())

            model.destroy()
            tf.keras.backend.clear_session()
            cv+=1

        bg_preds.append(preds)
        bg_truth.append(truth)
        bg_scores.append(score_matrix)

    print("\nFinished generating background...")
    print("\nBack ground run time: " + str(datetime.timedelta(seconds=(time.time() - bg_start_time))))         

    print("Saving background...")

    dirName = 'results/' + out + '/BG/'
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass

    bg_preds = np.array(bg_preds)
    bg_truth = np.array(bg_truth)
    bg_scores = np.array(bg_scores)
    bg_scores_mean = np.mean(np.array(bg_scores), axis=1)
    
    outfile = open(dirName + "bg_preds.pkl", "wb")
    pickle.dump(np.array(bg_preds), outfile)
    outfile.close()

    outfile = open(dirName + "bg_truth.pkl", "wb")
    pickle.dump(np.array(bg_truth), outfile)
    outfile.close()

    outfile = open(dirName + "bg_scores_mean.pkl", "wb")
    pickle.dump(bg_scores_mean, outfile)
    outfile.close() 

    bg_corr = []

    for i in range(0, bg_preds.shape[0]):
        for j in range(0,bg_preds.shape[-1]):
            p_vec = bg_preds[i,:,j]
            m_vec = bg_truth[i,:,j]
            cor = spearmanr(p_vec, m_vec)
            bg_corr.append(cor[0])

    outfile = open(dirName + "bg_correlations.pkl", "wb")
    pickle.dump(np.array(bg_corr), outfile)
    outfile.close()

    
###################################################
# Identify significantly correlated metabolites
################################################### 

if threshold == None:    
    cutoff_rho = np.quantile(bg_corr, 0.95)
else:
    cutoff_rho == threshold
    
print("The correlation cutoff is %.3f" % cutoff_rho)
print("%d of %d metabolites are significantly correlated" % (sum(correlation_cv_df["Mean"].values > cutoff_rho),
                                                             len(correlation_cv_df["Mean"].values)))

sig_metabolites = correlation_cv_df.index[correlation_cv_df["Mean"].values > cutoff_rho]

if annotation != None:
    annotation_df = pd.read_csv(annotation, index_col=0)
    annotated_metabolites = np.intersect1d(correlation_cv_df.index.values, annotation_df.index.values)
    sig_metabolites = annotated_metabolites[correlation_cv_df.loc[annotated_metabolites, "Mean"].values > cutoff_rho]
    
    print("%d of %d annotated metabolites are significantly correlated" % (len(sig_metabolites), len(annotated_metabolites)))
    
    barplot_df = pd.DataFrame(data={"Compound Name":annotation_df.loc[sig_metabolites, "Compound Name"].values, 
                                    "Spearman Correlation": correlation_cv_df.loc[sig_metabolites, "Mean"].values}, 
                              index=annotation_df.loc[sig_metabolites, "Compound Name"].values)

    barplot_df["Compound Name"] = [x.strip().capitalize() for x in barplot_df["Compound Name"].values]
    
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = fig.add_subplot(111)
    sns.barplot(x="Spearman Correlation", y='Compound Name', 
                data=barplot_df.groupby(barplot_df.index).max().sort_values(by="Spearman Correlation", ascending=False).head(20))
    plt.tight_layout()
    plt.savefig("results/" + out + "/Images/top_correlated_metabolites.png")

else:
    barplot_df = pd.DataFrame(data={"Compound Name": correlation_cv_df.loc[sig_metabolites].index.values, 
                                    "Spearman Correlation": correlation_cv_df.loc[sig_metabolites, "Mean"].values}, 
                              index=correlation_cv_df.loc[sig_metabolites].index.values)

    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = fig.add_subplot(111)
    sns.barplot(x="Spearman Correlation", y='Compound Name', 
                data=barplot_df.groupby(barplot_df.index).max().sort_values(by="Spearman Correlation", ascending=False).head(20))
    plt.savefig("results/" + out + "/Images/top_correlated_metabolites.png")

    
#######################################################
# Identify microbes with significant interaction scores
####################################################### 


mean_score_matrix = np.mean(np.array(score_matrices), axis=0)

reduced_mean_score_matrix = mean_score_matrix[:,[x in sig_metabolites for x in correlation_cv_df.index]]
reduced_bg_score_matrix = bg_scores_mean[:,:,[x in sig_metabolites for x in correlation_cv_df.index]]

sig_edge_matrix = np.zeros(reduced_mean_score_matrix.shape)

for mic in range(reduced_mean_score_matrix.shape[0]):
    for met in range(reduced_mean_score_matrix.shape[1]):
        sig_cutoff = np.abs(np.quantile(reduced_bg_score_matrix[:,mic,met], 0.975))
        if np.abs(reduced_mean_score_matrix[mic,met]) > sig_cutoff:
            sig_edge_matrix[mic,met]=1

            
sig_microbes = micro_comp_df.columns[np.sum(sig_edge_matrix, axis=1)> 0.01 * len(sig_metabolites)]

###################################################
# Compare Correlation Distributions
###################################################   

fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(111)
sns.distplot(bg_corr, label="Background")
sns.distplot(correlation_cv_df["Mean"].values, bins=20, label="Observed")
plt.axvline(x=cutoff_rho, color="red", lw=2, label="95% Cutoff")
plt.axvspan(cutoff_rho, 1.0, alpha=0.2, color='gray')

plt.title("Correlation Distributions for IBD", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xlabel("Spearman Correlation", fontsize=16)
plt.xlim(-1,1)
plt.text(0.85, 0.9,"Significant Region",
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.legend()
plt.savefig("results/" + out + "/Images/cv_bg_correlation_distributions.png")


score_matrix_df = pd.DataFrame(np.mean(score_matrices, axis=0), index=micro_comp_df.columns, 
                            columns=metab_comp_df.columns)
        
reduced_score_df = score_matrix_df.loc[sig_microbes,sig_metabolites]

binary_score_df = pd.DataFrame(np.clip(reduced_score_df/sig_cutoff, -1, 1), index=sig_microbes,
                               columns= sig_metabolites)

###################################################
# Compute Number of Microbial Modules
###################################################

mic_connectivity_matrices = {}

for i in range(2,20):
    mic_connectivity_matrices[i] = np.zeros((len(sig_microbes), len(sig_microbes)))

for s in score_matrices:
    mic_linkage_list = shc.linkage(np.clip(s[[x in sig_microbes for x in micro_comp_df.columns],:][:,[x in sig_metabolites for x in metab_comp_df.columns]]/sig_cutoff, -1,1), method='complete')

    for i in range(2,20):
        microbe_clusters = np.array(cut_tree(mic_linkage_list, n_clusters=i)).reshape(-1)
        one_hot_matrix = np.zeros((len(sig_microbes), i))
        for m in range(len(microbe_clusters)):
            one_hot_matrix[m, microbe_clusters[m]] = 1
        mic_connectivity_matrix = np.matmul(one_hot_matrix, np.transpose(one_hot_matrix)) 
        mic_connectivity_matrices[i] += mic_connectivity_matrix
        
fig = plt.figure(figsize=(8,4), dpi=300)
plt.subplot(1,2,1)
area_x = []
area_y = []
for i in range(2,20):
    consensus_matrix = mic_connectivity_matrices[i]/(num_run_cv * num_run)
    n = consensus_matrix.shape[0]
    consensus_cdf_x = []
    consensus_cdf_y = []
    area_x.append(int(i))
    
    prev_y = 0
    prev_x = 0
    area = 0
    for j in range(0,101):
        x = float(j)/100.0
        y = sum(sum(consensus_matrix <= x))/(n*(n-1))
        consensus_cdf_x.append(x)
        consensus_cdf_y.append(y)
        area += (x-prev_x) * (y)
        prev_x = x
    area_y.append(area)
    
    plt.plot(consensus_cdf_x, consensus_cdf_y, label=str(i) + " Clusters")

plt.xlabel("Consensus Index Value")
plt.ylabel("CDF")
plt.legend()

dk = []
for a in range(len(area_x)):
    if area_x[a] == 2:
        dk.append(area_y[a])
    else:
        dk.append((area_y[a] - area_y[a-1])/area_y[a-1])
        
plt.subplot(1,2,2)
plt.plot(area_x, dk, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Relative Increase of Area under CDF")
plt.axhline(0.025, linewidth=2, color='r')

num_microbiome_clusters = np.max(np.array(area_x)[np.array(dk) > 0.025])
print("Using %d Microbe Clusters" % num_microbiome_clusters)       
plt.savefig("results/" + out + "/Images/microbe_cluster_consensus.png")
      

###################################################
# Compute Number of Metabolic Modules
###################################################

met_connectivity_matrices = {}

for i in range(2,20):
    met_connectivity_matrices[i] = np.zeros((len(sig_metabolites), len(sig_metabolites)))

count = 0
for s in score_matrices:
    count += 1
    met_linkage_list = shc.linkage(np.transpose(np.clip(s[[x in sig_microbes for x in micro_comp_df.columns],:][:,[x in sig_metabolites for x in metab_comp_df.columns]]/sig_cutoff, -1,1)), method='complete')

    for i in range(2,20):
        metabolite_clusters = np.array(cut_tree(met_linkage_list, n_clusters=i)).reshape(-1)
        one_hot_matrix = np.zeros((len(sig_metabolites), i))
        for m in range(len(metabolite_clusters)):
            one_hot_matrix[m, metabolite_clusters[m]] = 1
        met_connectivity_matrix = np.matmul(one_hot_matrix, np.transpose(one_hot_matrix)) 
        met_connectivity_matrices[i] += met_connectivity_matrix
        
fig = plt.figure(figsize=(8,4), dpi=300)
plt.subplot(1,2,1)
area_x = []
area_y = []
for i in range(2,20):
    consensus_matrix = met_connectivity_matrices[i]/(num_run_cv * num_run)
    n = consensus_matrix.shape[0]
    consensus_cdf_x = []
    consensus_cdf_y = []
    area_x.append(int(i))
    
    prev_y = 0
    prev_x = 0
    area = 0
    for j in range(0,101):
        x = float(j)/100.0
        y = sum(sum(consensus_matrix <= x))/(n*(n-1))
        consensus_cdf_x.append(x)
        consensus_cdf_y.append(y)
        area += (x-prev_x) * (y)
        prev_x = x
    area_y.append(area)
    
    plt.plot(consensus_cdf_x, consensus_cdf_y, label=str(i) + " Clusters")

plt.xlabel("Consensus Index Value")
plt.ylabel("CDF")
plt.legend()

dk = []
for a in range(len(area_x)):
    if area_x[a] == 2:
        dk.append(area_y[a])
    else:
        dk.append((area_y[a] - area_y[a-1])/area_y[a-1])
        
plt.subplot(1,2,2)
plt.plot(area_x, dk, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Relative Increase of Area under CDF")
plt.axhline(0.02, linewidth=2, color='r')

num_metabolite_clusters = np.max(np.array(area_x)[np.array(dk) > 0.02])
print("Using %d Metabolite Clusters" % num_metabolite_clusters)       
plt.savefig("results/" + out + "/Images/metabolite_cluster_consensus.png")
      
        
###################################################
# Bicluster Interaction Matrix
###################################################

microbe_tree = shc.linkage(binary_score_df.values, method='complete')
metabolite_tree = shc.linkage(np.transpose(binary_score_df.values), method='complete')

metabolite_clusters = np.array(cut_tree(metabolite_tree, n_clusters=num_metabolite_clusters)).reshape(-1)
microbe_clusters = np.array(cut_tree(microbe_tree, n_clusters=num_microbiome_clusters)).reshape(-1)

metab_col_scale = 1/(num_metabolite_clusters-1)
micro_col_scale = 1/(num_microbiome_clusters-1)

micro_colors = [(0.5,1-x*micro_col_scale,x*micro_col_scale) for x in microbe_clusters]
metab_colors = [(1-x*metab_col_scale,0.5,x*metab_col_scale) for x in metabolite_clusters]

sns.clustermap(binary_score_df, method="complete", row_colors = micro_colors, col_colors=metab_colors, row_linkage=microbe_tree,
                col_linkage=metabolite_tree, cmap = "coolwarm", figsize=(8,8), cbar_pos=(0.05, 0.88, 0.025, 0.10))

plt.savefig("results/" + out + "/Images/" + "/clustermap.png", dpi=300)
        
reduced_score_df.to_csv("results/" + out + "/CV/" + "/interaction_score_matrix.csv")

micro_cluster_matrix = np.zeros((reduced_score_df.values.shape[0], reduced_score_df.values.shape[0]))
metab_cluster_matrix = np.zeros((reduced_score_df.values.shape[1], reduced_score_df.values.shape[1]))

for m in range(0, len(microbe_clusters)):
    for n in range(m, len(microbe_clusters)):
        if microbe_clusters[m] == microbe_clusters[n]:
            micro_cluster_matrix[m,n] = 1
            micro_cluster_matrix[n,m] = 1

for m in range(0, len(metabolite_clusters)):
    for n in range(m, len(metabolite_clusters)):
        if metabolite_clusters[m] == metabolite_clusters[n]:
            metab_cluster_matrix[m,n] = 1
            metab_cluster_matrix[n,m] = 1
                    
pd.DataFrame(data=micro_cluster_matrix, index=reduced_score_df.index, 
                columns=reduced_score_df.index).to_csv("results/" + out + "/CV/microbe_cluster_matrix.csv")   

pd.DataFrame(data=metab_cluster_matrix, index=reduced_score_df.columns, 
                columns=reduced_score_df.columns).to_csv("results/" + out + "/CV/metabolite_cluster_matrix.csv")  

metabolite_cluster_df = pd.DataFrame(data=metabolite_clusters, index=sig_metabolites, columns=["Cluster"])
microbe_cluster_df = pd.DataFrame(data=microbe_clusters, index=sig_microbes, columns=["Cluster"])

metabolite_cluster_df.to_csv("results/" + out + "/metabolite_clusters.csv")
microbe_cluster_df.to_csv("results/" + out +"/microbe_clusters.csv")



###################################################
# Determine Microbial Module Enrichment
################################################### 

if labels != None:
    try:
        labels_df=pd.read_csv(labels, index_col=0)
        label_set = np.unique(labels_df.values)
        g0 = samples[(labels_df.values==label_set[0]).reshape(-1)]
        g1 = samples[(labels_df.values==label_set[1]).reshape(-1)]

        micro_sub = micro_comp_df
        enriched_in = []
        p_list = []

        micro_comp_cluster_df = pd.DataFrame(index=samples)

        micro_sub = pd.DataFrame(index=micro_sub.index, columns=micro_sub.columns,
                                 data = micro_sub)


        for mc in range(num_microbiome_clusters):
            g0_cluster = micro_sub.loc[g0, microbe_cluster_df["Cluster"]==mc].mean(1).values
            g1_cluster =  micro_sub.loc[g1, microbe_cluster_df["Cluster"]==mc].mean(1).values
            micro_comp_cluster_df["Module " + str(mc + 1)] = micro_sub.loc[:,microbe_cluster_df["Cluster"]==mc].mean(1).values
            p_value = mannwhitneyu(g0_cluster, g1_cluster)[1]
            p_list.append(p_value)
            if p_value < 0.05:
                p_value_one_sided = mannwhitneyu(g0_cluster, g1_cluster, alternative="greater")[1]
                if p_value_one_sided < 0.05:
                    enriched_in.append(labels[0])
                else:
                    enriched_in.append(labels[1])
            else:
                enriched_in.append("None")

        micro_cluster_enrichment_df = pd.DataFrame(index=["Microbial Module " + str(x+1) for x in range(num_microbiome_clusters)])
        micro_cluster_enrichment_df["p-value"] = p_list
        micro_cluster_enrichment_df["Enriched"] = enriched_in
        micro_comp_cluster_df["Diagnosis"] = labels_df.values
        micro_cluster_enrichment_df.to_csv("results/" + out + "/microbiome_module_enrichment.csv")

        micro_box_df = pd.melt(micro_comp_cluster_df, id_vars= ["Diagnosis"], value_vars=micro_comp_cluster_df.columns[0:num_microbiome_clusters])
        plt.figure(figsize=(8,8), dpi=300)
        sns.boxplot(data=micro_box_df, x="variable", y="value", hue="Diagnosis")
        plt.xlabel("Microbiome Module")
        plt.ylabel("Mean Module Abundance")
        plt.title("Microbiome Module by Label")
        plt.savefig("results/" + out + "/Images/micro_module_enrichment.png")


        ###################################################
        # Determine Microbial Module Enrichment
        ################################################### 

        metab_sub = metab_comp_df
        enriched_in = []
        p_list = []

        metab_comp_cluster_df = pd.DataFrame(index=samples)
        metab_sub = pd.DataFrame(index=metab_sub.index, columns=metab_sub.columns,
                                 data = metab_sub)

        metab_sub = metab_sub[metabolite_cluster_df.index]


        for mc in range(num_metabolite_clusters):
            g0_cluster = metab_sub.loc[g0, metabolite_cluster_df["Cluster"]==mc].mean(1).values
            g1_cluster =  metab_sub.loc[g1, metabolite_cluster_df["Cluster"]==mc].mean(1).values
            metab_comp_cluster_df["Module " + str(mc + 1)] = metab_sub.loc[:,metabolite_cluster_df["Cluster"]==mc].mean(1).values
            p_value = mannwhitneyu(g0_cluster, g1_cluster)[1]
            p_list.append(p_value)
            if p_value < 0.05:
                p_value_one_sided = mannwhitneyu(g0_cluster, g1_cluster, alternative="greater")[1]
                if p_value_one_sided < 0.05:
                    enriched_in.append(labels[0])
                else:
                    enriched_in.append(labels[1])
            else:
                enriched_in.append("None")


        metab_cluster_enrichment_df = pd.DataFrame(index=["Metabolite Module " + str(x+1) for x in range(num_metabolite_clusters)])
        metab_cluster_enrichment_df["p-value"] = p_list
        metab_cluster_enrichment_df["Enriched"] = enriched_in
        metab_comp_cluster_df["Diagnosis"] = labels_df.values
        metab_cluster_enrichment_df.to_csv("results/" + out + "/metabolite_cluster_enrichment.csv")

        metab_box_df = pd.melt(metab_comp_cluster_df, id_vars= ["Diagnosis"], value_vars=metab_comp_cluster_df.columns[0:num_metabolite_clusters])
        plt.figure(figsize=(8,8), dpi=300)
        sns.boxplot(data=metab_box_df, x="variable", y="value", hue="Diagnosis")
        plt.xlabel("Metabolite Module")
        plt.ylabel("Mean Module Abundance")
        plt.title("Metabolite Module by Label")
        plt.savefig("results/" + out + "/Images/metab_module_enrichment.png")
    except:
        print("Warning! Could not open label file and perform module enrichment!")
        
###################################################
# Train Ensemble of Neural Networks on Full Dataset
###################################################  

# Set up output directory for training on full dataset
dirName = 'results/' + out + '/Full'
    
try:
    os.mkdir(dirName)
except FileExistsError:
    pass
              
microbe_cluster_matrix_list = []
metabolite_cluster_matrix_list = []

for run in range(0,num_run):

    # Set up output directory for training on full dataset
    dirName = 'results/' + out + '/Full/' + str(run) 
    
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass
                
    train_micro = micro_comp_df
    train_metab = metab_comp_df
    
    # Log transform data if RA
    if micro_norm == "RA" or micro_norm == None:
        train_micro = np.log(train_micro + 1)            
                
    if metab_norm == "RA" or metab_norm == None:
        train_metab = np.log(train_metab + 1)
            
    # Scale data before neural network training
    micro_scaler = StandardScaler().fit(train_micro)
    train_micro = micro_scaler.transform(train_micro)
                
    metab_scaler = StandardScaler().fit(train_metab)
    train_metab = metab_scaler.transform(train_metab)

    # Aggregate paired microbiome and metabolomic data
    train = (train_micro, train_metab)

    print("Run: %02d" % (run + 1), end="\r")
                
    # Construct Neural Network Model
    model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=l1, l2=l2, 
                        num_layer=num_layer, layer_nodes=layer_nodes, dropout=dropout)

    #Train Neural Network Model
    model.train(train)
          
    score_matrix_df = pd.DataFrame(data=model.get_scores(), index=micro_comp_df.columns, 
                                   columns=metab_comp_df.columns)
    score_matrix_df.to_csv(dirName + "/score_matrix.csv")
    
    reduced_score_df = score_matrix_df.loc[sig_microbes,sig_metabolites]
 
    metabolite_tree = shc.linkage(np.transpose(np.clip(s[[x in sig_microbes for x in micro_comp_df.columns],:][:,[x in sig_metabolites for x in metab_comp_df.columns]]/sig_cutoff, -1,1)), method='complete')
    
    microbe_tree = shc.linkage(np.clip(s[[x in sig_microbes for x in micro_comp_df.columns],:][:,[x in sig_metabolites for x in metab_comp_df.columns]]/sig_cutoff, -1,1), method='complete')

    metabolite_clusters = np.array(cut_tree(metabolite_tree, n_clusters=num_metabolite_clusters)).reshape(-1)
    microbe_clusters = np.array(cut_tree(microbe_tree, n_clusters=num_microbiome_clusters)).reshape(-1)
    
    metab_col_scale = 1/(num_metabolite_clusters-1)
    micro_col_scale = 1/(num_microbiome_clusters-1)
    
    micro_colors = [(0.5,1-x*micro_col_scale,x*micro_col_scale) for x in microbe_clusters]
    metab_colors = [(1-x*metab_col_scale,0.5,x*metab_col_scale) for x in metabolite_clusters]
    
    sns.clustermap(reduced_score_df, method="complete", row_colors = micro_colors, col_colors=metab_colors, 
                   cmap = "coolwarm", figsize=(8,8), cbar_pos=(0.05, 0.88, 0.025, 0.10))
    plt.savefig(dirName + "/clustermap.png", dpi=300)
    reduced_score_df.to_csv(dirName + "/interaction_score_matrix.csv")

    micro_cluster_matrix = np.zeros((reduced_score_df.values.shape[0], reduced_score_df.values.shape[0]))
    metab_cluster_matrix = np.zeros((reduced_score_df.values.shape[1], reduced_score_df.values.shape[1]))
    
    for m in range(0, len(microbe_clusters)):
        for n in range(m, len(microbe_clusters)):
            if microbe_clusters[m] == microbe_clusters[n]:
                micro_cluster_matrix[m,n] = 1
                micro_cluster_matrix[n,m] = 1

    for m in range(0, len(metabolite_clusters)):
        for n in range(m, len(metabolite_clusters)):
            if metabolite_clusters[m] == metabolite_clusters[n]:
                metab_cluster_matrix[m,n] = 1
                metab_cluster_matrix[n,m] = 1
    pd.DataFrame(data=micro_cluster_matrix, index=reduced_score_df.index, 
                 columns=reduced_score_df.index).to_csv(dirName + "/microbe_cluster_matrix.csv")   
    
    pd.DataFrame(data=metab_cluster_matrix, index=reduced_score_df.columns, 
                 columns=reduced_score_df.columns).to_csv(dirName + "/metabolite_cluster_matrix.csv")  
    
    microbe_cluster_matrix_list.append(micro_cluster_matrix)
    metabolite_cluster_matrix_list.append(metab_cluster_matrix)  
    model.model.save(dirName + "/network_model.h5")

    model.destroy()
    tf.keras.backend.clear_session()
    

consensus_list = []
micro_cluster_matrix_df = pd.read_csv("results/" + out + "/CV/microbe_cluster_matrix.csv", index_col=0)

for r in range(num_run):
    run_micro_cluster_matrix_df = pd.read_csv("results/" + out + "/Full/" + str(r) + "/microbe_cluster_matrix.csv", index_col=0)
    hits = np.sum(micro_cluster_matrix_df.values == run_micro_cluster_matrix_df.values)
    total = micro_cluster_matrix_df.values.shape[0] * micro_cluster_matrix_df.values.shape[1]
    consensus_list.append(hits/total)
    
print("Selecting Full Model %d with consensus %.2f" % (np.argmax(consensus_list), np.max(consensus_list)))

final_model = tf.keras.models.load_model("results/" + out + "/Full/" + str(np.argmax(consensus_list)) + "/network_model.h5")
final_model.save("results/" + out + "/final_network_model.h5")

if external_micro != None:
    test_micro = external_micro_comp_df

    if micro_norm == "RA" or micro_norm == None:
        test_micro = np.log(test_micro + 1)
    
    test_micro = micro_scaler.transform(test_micro)
    pred = final_model.predict(test_micro)
    inv_pred = metab_scaler.inverse_transform(pred)
    if metab_norm == "RA" or metab_norm == None:
        inv_pred = np.exp(inv_pred) - 1
        inv_pred = inv_pred/np.sum(inv_pred)
        
    external_pred_df = pd.DataFrame(data = inv_pred, index=external_samples, columns=metab_comp_df.columns)
    external_pred_df.to_csv("results/" + out + "/external_predictions.csv")
    
if external_metab != None:
    external_corr = external_metab_comp_df.corrwith(external_pred_df, method="spearman")
    print("External mean correlation %.2f" % (np.mean(external_corr)))
    print("%d of %d metabolites are significantly correlated in external evaluation" % (sum(external_corr.values > cutoff_rho),
                                                                                        len(correlation_cv_df.values)))  
       
    external_sig_metabolites = annotated_metabolites[external_corr.loc[annotated_metabolites].values > cutoff_rho]
    
    print("%d of %d annotated metabolites are significantly correlated in the external evaluation" % (len(external_sig_metabolites),
                                                                                                      len(annotated_metabolites)))