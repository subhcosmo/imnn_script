# imnn_script
This is a code developed by me for my Master's thesis that tried to combine Fisher maximising IMNN compression with the SCRIPT semi-numerical code of reionization, and then further using both SBI and MCMC for parameter inference.
Each folder has all the 5 techniques used to do compression and then parameter inference named: IMNN+SBI, 1DPS+SBI, 1DPS+MCMC, 2DPS+SBI and 2DPS+MCMC
The target files and the posterior files are also present in the corresponding folder. The covariance matrix are also supplied (along with the code to calculate it, however you would need to have the 1000+1000 corresponding PS to get to the covariance matrix)
The trained NN is also supplied to do the inference using SBI.
