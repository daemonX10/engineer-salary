starting enhanced model training...
training data shape: (1280, 317)
testing data shape: (854, 316)
performing exploratory analysis...
target distribution:
salary_category
high      501
low       419
medium    360
name: count, dtype: int64
target distribution by job_state:
salary_category  high   low  medium
job_state
ak               0.00  0.00    1.00
al               0.00  1.00    0.00
ar               0.00  1.00    0.00
az               0.00  0.50    0.50
ca               0.40  0.42    0.18
co               0.38  0.12    0.50
ct               1.00  0.00    0.00
dc               0.25  0.12    0.62
fl               0.00  0.50    0.50
ga               0.50  0.40    0.10
ia               0.00  0.00    1.00
il               0.20  0.47    0.33
in               0.00  0.50    0.50
ky               1.00  0.00    0.00
la               1.00  0.00    0.00
ma               0.50  0.21    0.29
md               0.40  0.00    0.60
mi               0.00  0.50    0.50
mn               0.14  0.29    0.57
mo               1.00  0.00    0.00
nc               0.40  0.60    0.00
nj               0.30  0.30    0.40
nm               0.00  0.50    0.50
nv               0.00  1.00    0.00
ny               0.39  0.24    0.37
oh               0.25  0.50    0.25
ok               0.00  0.00    1.00
or               0.25  0.00    0.75
pa               0.50  0.25    0.25
sc               0.33  0.33    0.33
sd               0.00  1.00    0.00
tn               0.00  1.00    0.00
tx               0.32  0.36    0.32
ut               0.00  1.00    0.00
va               0.53  0.21    0.26
wa               0.38  0.46    0.15
target distribution by feature_1:
salary_category  high  low  medium
feature_1
a                0.41  0.3    0.29
b                0.00  1.0    0.00
c                0.00  1.0    0.00
d                0.20  0.6    0.20
e                0.00  1.0    0.00
correlation of feature_3 with target:
feature_3
false            0.37  0.35    0.28
true             0.55  0.14    0.31
correlation of feature_4 with target:
feature_4
false            0.39  0.34    0.27
true             0.36  0.10    0.54
correlation of feature_5 with target:
feature_5
false            0.37  0.37    0.26
true             0.48  0.13    0.38
correlation of feature_6 with target:
feature_6
false            0.35  0.39    0.26
true             0.41  0.30    0.29
correlation of feature_7 with target:
feature_7
false            0.39  0.34    0.26
true             0.39  0.31    0.31
correlation of feature_8 with target:
feature_8
false            0.45  0.26    0.29
true             0.12  0.63    0.24
correlation of feature_10 with target:
feature_10
6.0              0.00  1.00    0.00
12.0             0.24  0.34    0.42
24.0             0.29  0.34    0.37
36.0             0.35  0.33    0.32
48.0             0.42  0.34    0.24
60.0             0.51  0.24    0.24
72.0             0.59  0.20    0.20
84.0             0.60  0.26    0.15
96.0             0.70  0.18    0.12
108.0            0.86  0.00    0.14
120.0            0.46  0.41    0.14
132.0            1.00  0.00    0.00
144.0            0.83  0.17    0.00
168.0            1.00  0.00    0.00
180.0            0.50  0.33    0.17
192.0            0.50  0.50    0.00
240.0            0.00  1.00    0.00
300.0            0.00  0.00    1.00
correlation of feature_11 with target:
feature_11
false            0.14  0.58    0.28
true             0.40  0.32    0.28
warning: error in feature_9 binning: numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
processed training data shape: (1280, 176)
processed test data shape: (854, 182)
0    501
1    419
2    360
training and evaluating multiple model architectures...
training randomforest...
randomforest - cv score: 0.7414, validation score: 0.7461 (time: 4.07s)
training xgboost...
xgboost - cv score: 0.7453, validation score: 0.7617 (time: 14.41s)
training lightgbm...
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003188 seconds.
you can set `force_col_wise=true` to remove the overhead.
[lightgbm] [info] total bins 12794
[lightgbm] [info] number of data points in the train set: 1024, number of used features: 130
[lightgbm] [info] start training from score -0.940007
[lightgbm] [info] start training from score -1.114361
[lightgbm] [info] start training from score -1.268511
[lightgbm] [warning] no further splits with positive gain, best gain: -inf
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002931 seconds.
[lightgbm] [info] total bins 12798
[lightgbm] [info] number of data points in the train set: 1024, number of used features: 131
[lightgbm] [info] start training from score -0.937510
[lightgbm] [info] start training from score -1.117341
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003091 seconds.
[lightgbm] [info] total bins 12820
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002790 seconds.
[lightgbm] [info] total bins 12808
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003080 seconds.
[lightgbm] [info] total bins 12819
[lightgbm] [info] number of data points in the train set: 1024, number of used features: 129
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003067 seconds.
[lightgbm] [info] total bins 12810
lightgbm - cv score: 0.7414, validation score: 0.7500 (time: 14.20s)
training catboost...
catboost - cv score: 0.7477, validation score: 0.7227 (time: 119.01s)
training gradientboosting...
gradientboosting - cv score: 0.7477, validation score: 0.7773 (time: 193.42s)
best base model: gradientboosting with validation score 0.7773
creating ensemble models...
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002483 seconds.
voting ensemble - validation score: 0.7734
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002372 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001822 seconds.
[lightgbm] [info] total bins 10594
[lightgbm] [info] number of data points in the train set: 682, number of used features: 126
[lightgbm] [info] start training from score -0.937781
[lightgbm] [info] start training from score -1.117858
[lightgbm] [info] start training from score -1.267534
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002546 seconds.
[lightgbm] [info] total bins 10734
[lightgbm] [info] number of data points in the train set: 683, number of used features: 127
[lightgbm] [info] start training from score -0.935508
[lightgbm] [info] start training from score -1.119323
[lightgbm] [info] start training from score -1.268999
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002242 seconds.
[lightgbm] [info] total bins 10640
[lightgbm] [info] start training from score -0.939246
[lightgbm] [info] start training from score -1.114849
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.000551 seconds.
[lightgbm] [info] total bins 3805
[lightgbm] [info] number of data points in the train set: 1024, number of used features: 15
stacking ensemble - validation score: 0.7148
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002916 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002805 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001895 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001631 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002993 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.000750 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002449 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001752 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.007073 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.008034 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001628 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001584 seconds.
[lightgbm] [info] total bins 7117
[lightgbm] [info] number of data points in the train set: 454, number of used features: 115
[lightgbm] [info] start training from score -0.936314
[lightgbm] [info] start training from score -1.120885
[lightgbm] [info] start training from score -1.266067
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001755 seconds.
[lightgbm] [info] total bins 7131
[lightgbm] [info] number of data points in the train set: 455, number of used features: 116
[lightgbm] [info] start training from score -0.938514
[lightgbm] [info] start training from score -1.116351
[lightgbm] [info] start training from score -1.268267
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001787 seconds.
[lightgbm] [info] total bins 7032
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.000431 seconds.
[lightgbm] [info] total bins 3383
[lightgbm] [info] number of data points in the train set: 682, number of used features: 15
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002015 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001501 seconds.
[lightgbm] [info] total bins 7136
[lightgbm] [info] start training from score -0.932912
[lightgbm] [info] start training from score -1.123085
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001569 seconds.
[lightgbm] [info] total bins 7277
[lightgbm] [info] number of data points in the train set: 455, number of used features: 118
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001776 seconds.
[lightgbm] [info] total bins 7224
[lightgbm] [info] number of data points in the train set: 456, number of used features: 118
[lightgbm] [info] start training from score -0.935107
[lightgbm] [info] start training from score -1.118547
[lightgbm] [info] start training from score -1.270463
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.000157 seconds.
[lightgbm] [info] total bins 3397
[lightgbm] [info] number of data points in the train set: 683, number of used features: 15
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001580 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.001531 seconds.
[lightgbm] [info] total bins 7053
[lightgbm] [info] number of data points in the train set: 455, number of used features: 117
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002092 seconds.
[lightgbm] [info] total bins 7218
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.002205 seconds.
[lightgbm] [info] total bins 7174
[lightgbm] [info] number of data points in the train set: 456, number of used features: 121
[lightgbm] [info] start training from score -0.940709
[lightgbm] [info] start training from score -1.111858
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.000248 seconds.
[lightgbm] [info] total bins 3398
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003030 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003558 seconds.
[lightgbm] [info] auto-choosing col-wise multi-threading, the overhead of testing was 0.003498 seconds.
enhanced stacking - validation score: 0.7344
selected gradientboosting as best model
top 20 feature importances:
feature_2_9_ratio       0.057192
feature_2_cubed         0.043136
feature_2               0.041482
feat1_a                 0.036722
job_desc_pca_1          0.035980
feature_2_squared       0.031099
job_desc_svd_1          0.030833
feature_2_sqrt          0.030799
feature_2_log           0.029887
job_title_cb_encoded    0.029126
job_desc_pca_4          0.026941
feature_10_8_diff       0.026442
job_title_encoded       0.026338
job_desc_svd_4          0.024499
job_recency             0.021680
job_desc_median         0.017382
job_desc_q25            0.014577
job_desc_min            0.013077
job_desc_svd_6          0.012906
job_desc_svd_14         0.012615
dtype: float64
training final model on full dataset...
model saved as 'best_model.joblib'
predictions saved to data/solution_format.csv
prediction distribution: {'high': 313, 'low': 279, 'medium': 262}
total execution time: 1800.17 seconds (30.00 minutes)