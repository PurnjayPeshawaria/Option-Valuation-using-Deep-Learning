# COMP9417-project

## Section 2.2 Hyper-Parameter Training
In the paper, section 2.2 shows the hyper parameters that were chosen for the artificial neural network after doing random selection from the hyper-parameter selection options/ranges. The hyper parameter tuning is in the file `ANN_Tune_Hyper_Parameter.py`.

## Section 3.2 Finding the best learning rate
Finding the best learning rate was determined via training the model using different learning rates and finding the learning rates which yields the steepest gradient. This is shown in the file `ANN_find_best_learning_rate.py`.

## Section 4 The Heston Model
The code for approximating option prices using the COS method with the Heston model, and using those prices to train an ANN are in the file `heston.py`.

## Section 5 Implied Volatility
Calculating the implied volatility of the option using Brent's method and ANN is in the file `impliedVolatility.py`.

## Section 6 Finite Difference Methods for Valuation of Options
All the files that can be shown for finite difference methods for valuing options can be seen in files `FDM_results.py`, `black_scholes_unscaled.py`, `bspdeCN.py`, `bspdeexp.py`, `bspdeimp.py` and `compare_FDM_ANN.py`.

