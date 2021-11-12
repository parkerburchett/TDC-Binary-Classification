# TDC-Binary-Classification

This is a repository that contains the work I did to build a model that predicts if a molecule will inhibit an enzyme based purely on its chemical structure.  The model is currently the best on this dataset and the embedding algorithm I wrote is applicable to other molecule classification/regression tasks. 

I wrote a custom embedding algorithm called ColorRefinement that embeds variable sized molecules into a feature space such that it preserves subgraph attribute and topology.

Check out how the embedding algorithm works here:  https://github.com/parkerburchett/TDC-Binary-Classification/blob/main/ColorRefinement/ColorRefinement%20Tutorial.ipynb 

You can read the final report https://github.com/parkerburchett/TDC-Binary-Classification/blob/main/Final%20Report.pdf for a high level summary of the methodology and outcomes.    

That embedding algorithm + a weighted ensemble of LGBM models got me to 1st place on the leaderboard! https://tdcommons.ai/benchmark/admet_group/10cyp2c9i/  

Older repo containing the different approaches I tried.  https://github.com/parkerburchett/TDC-DeepLearning 
