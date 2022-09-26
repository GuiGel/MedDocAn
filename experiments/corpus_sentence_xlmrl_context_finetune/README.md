# XLM Roberta Large Results

After a grid search process, we run the training 3 times with the best hyperparameters found and report the average **$F_1$** scores and the corresponding standard deviation. 

The model is the following:

TODO

## Results

We evaluate the final model on the **dev** and **test** sets with the *Subtrack 1** for NER and **Subtrack 2** for the spans in the strict and merged version.


DEV

Report (SYSTEM: brat):
------------------------------------------------------------
SubTrack 1 [NER]                   Measure        Micro               
------------------------------------------------------------
Total (250 docs)                   Leak           NA                  
                                   Precision      0.971696 ± 0.000807  
                                   Recall         0.978452 ± 0.001130  
                                   F1             0.975062 ± 0.000886  
------------------------------------------------------------
