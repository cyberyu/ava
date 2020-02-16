# AVA
---


## Project repo for "Yu et al., AVA: A Financial Service Chatbot based on Deep BidirectionalTransformer, submitted to ACM KDD 2020" 

Welcome!

## Table of Contents

- [Additional Results for the Main Paper](#additional-results-for-the-main-paper)
  - [Histogram of Uncertainties by Dropout Ratios](#histogram-of-uncertainties-by-dropout-ratios)
  - [Uncertainty and Accuracy Curves by Dropout Ratios 5-class vs 381-class](#uncertainty-and-accuracy-curves-by-dropout-ratios-5-class-vs-381-class)
  - [Grid Search for Optimal Threshold on Dropout](#grid-search-for-optimal-threshold-on-dropout)  
  - [Optimal Threshold Learning on Dropout 381 classes](#optimal-threshold-learning-on-dropout-381-classes)  

- [BERT Model Scripts](#bert-model-scripts)
  
- [Threshold Optimization Scripts](#threshold-optimization-scripts)

- [Sentence Completion Model Scripts](#sentence-completion-model-scripts)
- [RASA scripts for chatbot](#rasa-scripts-for-chatbot)
- [License](#license)
- [Contributors](#contributors)

## Additional Results for the Main Paper

##### Histogram of Uncertainties by Dropout Ratios
We list out two columns, nine rows of histograms visualizing difference of three distributions (training, test, irrelevant questions).  The left column shows models trained for 10 epochs and the right are models trained for 30 epochs. From 1st row to 9th row we show distributions obtained from dropout at 10, 20, 30, ..., 90 percent.  

<img src="images/original/dropout_10percent_10iterations.png" width="350"/> <img src="images/original/dropout_10percent_30iterations.png" width="350"  />
<img src="images/original/dropout_20percent_10iterations.png" width="350"/> <img src="images/original/dropout_20percent_30iterations.png" width="350"  />
<img src="images/original/dropout_30percent_10iterations.png" width="350"/> <img src="images/original/dropout_30percent_30iterations.png" width="350"  />
<img src="images/original/dropout_40percent_10iterations.png" width="350"/> <img src="images/original/dropout_40percent_30iterations.png" width="350"  />
<img src="images/original/dropout_50percent_10iterations.png" width="350"/> <img src="images/original/dropout_50percent_30iterations.png" width="350"  />
<img src="images/original/dropout_60percent_10iterations.png" width="350"/> <img src="images/original/dropout_60percent_30iterations.png" width="350"  />
<img src="images/original/dropout_70percent_10iterations.png" width="350"/> <img src="images/original/dropout_70percent_30iterations.png" width="350"  />
<img src="images/original/dropout_80percent_10iterations.png" width="350"/> <img src="images/original/dropout_80percent_30iterations.png" width="350"  />
<img src="images/original/dropout_90percent_10iterations.png" width="350"/> <img src="images/original/dropout_90percent_30iterations.png" width="350"  />

##### Uncertainty and Accuracy Curves by Dropout Ratios 5-class vs 381-class
The 5-class model is trained using the same dataset mentioned in the main paper, but only using Tier-1 classes. 

Uncertainty and Accuracy curves obtained on 381-class model. <br/>
<img src="images/original/acc_dropout_10epoch.png" width="250"/>  <img src="images/original/acc_dropout_30epoch.png" width="250" />  <img src="images/original/epoch_compare.png" width="250" />
<br/>

Uncertainty and Accuracy curves obtained on 5-class model.
<br/>
<img src="images/original/acc_dropout_10epoch_5class.png" width="250"/><img src="images/original/acc_dropout_30epoch_5class.png" width="250"/><img src="images/original/epoch_compare_5class.png" width="250" />

##### Grid Search For Optimal Threshold on Dropout
Left side is grid search for 381-class problem.  Right side is grid search for 5-class problem.
<img src="images/original/bfsearch.png" width="350"/>  <img src="images/original/bfsearch_5class.png" width="350"/>

##### Optimal threshold learning on dropout 381 classes

We did more experiments using different \delta values to optimizing the thresholds from Dropout, and results can be found here.

##### Uncertainty comparison 5-class vs 381-class

<img src="images/original/dropout_381_d5_10iter_10pct.png" width="650"/>
<br/>
<img src="images/original/dropout_5_d5_10iter_10pct.png" width="650"/>

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

Released 2020 by [Shi Yu](https://github.com/cyberyu) 
