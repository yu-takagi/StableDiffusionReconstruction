# Features and DecodedFeatures

bdpy provides classes to handle DNN's (true) features and decoded features: `dataform.Features` and `dataform.DecodedFeatures`.

## Basic usage

``` python
from bdpy.dataform import Features, DecodedFeatures


## Initialize

features = Features('/path/to/features/dir')

decoded_features = DecodedFeatures('/path/to/decoded/features/dir')

## Get features as an array

feat = features.get(layer='conv1')

decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC', label='stimulus-0001)  # Decoded features for specified sample (label)
decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC')                        # Decoded features from all avaiable samples

# Decoded features with CV
decfeat = decoded_features.get(layer='conv1', subject='sub-01', roi='VC', fold='cv_fold1)

## List labels

feat_labels = features.labels

decfeat_labels = decoded_features.labels           # All available labels
decfeat_labels = decoded_features.selected_labels  # Labels assigned to decoded features previously obtained by `get` method
```

## Feature statistics

``` python
features.statistic('mean', layer='fc8')
features.statistic('std', layer='fc8')          # Default ddof = 1
features.statistic('std, ddof=0', layer='fc8')

decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC')
decoded_features.statistic('std', layer='fc8', subject='sub-01', roi='VC')          # Default ddof = 1
decoded_features.statistic('std, ddof=0', layer='fc8', subject='sub-01', roi='VC')

# Decoded features with CV
decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC', fold='cv_fold1')  # Mean within the specified fold
decoded_features.statistic('mean', layer='fc8', subject='sub-01', roi='VC')

# If `fold` is omitted for CV decoded features, decoded features are pooled across add CV folds and then the statistics are calculated.

```


