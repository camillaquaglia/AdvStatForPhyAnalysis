---
title: "COVID-19 vaccine efficacies"
subtitle: "An Inference Problem"
author: "Camilla Quaglia"
date: "June 8th 2021"
output:
  html_document:
    keep_md: true
    toc: true
    number_sections: true
    toc_float: 
        collapsed: false
---



# Introduction 
<font size="3">

- Our goal: obtain the probability distribution of the efficacy
- Tools: Bayesian Network + Markov Chain Monte Carlo 
</font>
<br />
<br />
From Moderna’s press release (November 16, 2020): *" This first interim analysis was based on 95 cases, of which 90 cases of COVID-19 were observed in the placebo group versus 5 cases observed in the mRNA-1273 group, resulting in a <span style="color: red;"> point estimate of vaccine efficacy of 94.5% (p <0.0001)</span>"*
<br />
<br />

![](fake.PNG){width=60%}
<br />
<br />

![](fake2.PNG){width=60%}

# Set up the model
## Bayesian Network
![](n.PNG){width=50%}

- $n_{V}$ = number of individuals in the vaccine group
- $n_{P}$ = number of individuals in the placebo group
- $n_{V_{I}}$ and $n_{P_{I}}$ = number of infected individual, in the two groups, after the trial period
- $p_{A}$ = assault probability
- $n_{V_{A}}$ and $n_{P_{A}}$ = number of assaulted individuals in each group
<br />
<br />
<font size= "3">
The assaulted individuals of the vaccinated group are protected by the vaccine. If $\epsilon$ is the **efficacy** of the vaccine, the probability for them of being infected is equal to 1 − $\epsilon$ .
<br />
<br />
<center>
$n_{V_{A}} \sim Binom(n_{V},p_{A})$
<br />
<br />
$n_{P_{A}} \sim Binom(n_{P},p_{A})$
<br />
<br />
$n_{V_{I}} \sim Binom(n_{V_{A}},1-\epsilon)$
</center>
<br />
<br />
The calculation is done through a MCMC with the help of the program `JAGS`, that is interfaced with `R` via the package `rjags`


```r
library(rjags)
```

```
## Loading required package: coda
```

```
## Linked to JAGS 4.3.0
```

```
## Loaded modules: basemod,bugs
```
## BUGS model
The model written in `BUGS` language is: 

```r
# DEFINE the MODEL 
model_vaccine <- "model{
nP.I ~ dbin(pA, nP)         # dbinom() in R 
nV.A ~ dbin(pA, nV) 
nV.I ~ dbin(prob, nV.A)     # prob = 1 - epsilon

# Priors of nodes with no parents 
pA ~ dbeta(1,1) 
prob ~ dbeta(1,1) 

epsilon <- 1 - prob         # the quantity we want to trace in the chain 
}"
```
The priors both for the $p_{A}$ and $prob$ (= the probability that an assaulted vaccinated person being infected) are **uniform priors**.

# Data
<center>
Moderna: $n_{V_{I}} = 5$ $~$$~$$~$ $n_{P_{I}}=90$
<br />
Pfizer:  $n_{V_{I}} = 8$ $~$$~$$~$ $n_{P_{I}}=162$
</center>
<br />
What about $n_{V}$ and $n_{P}$ ? The exact number is not critical at all in regard to the value of efficacy (if they are enough larger than $n_{V_{I}}$ and $n_{P_{I}}$)


# Analysis and Results
The model is red using the `jags.model` function. The latter creates an object of the class `jags`.

$~$

```r
# function to COMPILE THE MODEL

model_jags <- function(data) {
  m_j <- jags.model(textConnection(model_vaccine),data=data,inits = list(.RNG.name= "base::Wichmann-Hill", .RNG.seed=123))
  return(m_j)                                             
}

# DATA
Moderna <- list(nV.I=5,nP.I=90,nV=15000,nP=15000)
Pfizer <- list(nV.I=8,nP.I=162,nV=15000,nP=15000)

model_jags_Moderna <- model_jags(Moderna)
```

```
## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 2
##    Unobserved stochastic nodes: 3
##    Total graph size: 9
## 
## Initializing model
```

```r
model_jags_Pfizer <- model_jags(Pfizer)
```

```
## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 2
##    Unobserved stochastic nodes: 3
##    Total graph size: 9
## 
## Initializing model
```


```r
# Update the Markov chain (Burn-in)
update(model_jags_Moderna, 1000) 
update(model_jags_Pfizer, 1000) 
```
Samples from the model are extracted using the `coda.samples` function. The latter creates an object of the class `mcmc.list`, that is a *summary* of the posterior distribution.
<br />
<br />
Arguments: `model` = our model, `variable.names` = values we want to trace in the chain, `n.iter` = steps to run the chain

```r
# SIMULATE the posterior -> draw 100'000 approximated samples from the posterior
draw.posterior <- function(model){
  sim <- coda.samples(model=model,variable.names = c("epsilon","nV.A"),n.iter=100000)
  return(sim)
} 
```


```r
posterior_moderna <- draw.posterior(model_jags_Moderna)
Summary_Moderna <- summary(posterior_moderna)
Summary_Moderna
```

```
## 
## Iterations = 2001:102000
## Thinning interval = 1 
## Number of chains = 1 
## Sample size per chain = 1e+05 
## 
## 1. Empirical mean and standard deviation for each variable,
##    plus standard error of the mean:
## 
##            Mean       SD  Naive SE Time-series SE
## epsilon  0.9327  0.02856 9.031e-05      0.0001141
## nV.A    89.0975 13.44168 4.251e-02      0.0853848
## 
## 2. Quantiles for each variable:
## 
##            2.5%     25%     50%     75%    97.5%
## epsilon  0.8659  0.9165  0.9368  0.9535   0.9759
## nV.A    64.0000 80.0000 89.0000 98.0000 117.0000
```

```r
posterior_pfizer <- draw.posterior(model_jags_Pfizer)
Summary_Pfizer <- summary(posterior_pfizer)
Summary_Pfizer
```

```
## 
## Iterations = 2001:102000
## Thinning interval = 1 
## Number of chains = 1 
## Sample size per chain = 1e+05 
## 
## 1. Empirical mean and standard deviation for each variable,
##    plus standard error of the mean:
## 
##             Mean       SD  Naive SE Time-series SE
## epsilon   0.9442  0.01911 6.044e-05      7.392e-05
## nV.A    161.1126 17.92059 5.667e-02      1.100e-01
## 
## 2. Quantiles for each variable:
## 
##             2.5%      25%      50%      75%    97.5%
## epsilon   0.9008   0.9329   0.9464   0.9579   0.9751
## nV.A    128.0000 149.0000 161.0000 173.0000 198.0000
```


```r
Summary_Moderna$statistics
```

```
##               Mean          SD     Naive SE Time-series SE
## epsilon  0.9327322  0.02855784 9.030781e-05    0.000114056
## nV.A    89.0974700 13.44167908 4.250632e-02    0.085384769
```


```r
epsilonM_mean <- Summary_Moderna$statistics[1,1]
epsilonM_std <- Summary_Moderna$statistics[1,2]
mean.nV.A_M <- Summary_Moderna$statistics[2,1]
std.nV.A_M <- Summary_Moderna$statistics[2,2]

Summary_Pfizer$statistics
```

```
##                Mean          SD     Naive SE Time-series SE
## epsilon   0.9442151  0.01911148 6.043582e-05   7.391571e-05
## nV.A    161.1126000 17.92059409 5.666989e-02   1.099835e-01
```


```r
epsilonP_mean <- Summary_Pfizer$statistics[1,1]
epsilonP_std <- Summary_Pfizer$statistics[1,2]
mean.nV.A_P <- Summary_Pfizer$statistics[2,1]
std.nV.A_P <- Summary_Pfizer$statistics[2,2]
```

In the following, plots for the simulated posterior.

```r
plot(posterior_moderna[,1],trace=TRUE,col="navy",main =  ~ epsilon ~ "Moderna's vaccine",cex.lab=1.5, cex.axis=1.5, cex.main=1.6 )
```

![](R_project_files/figure-html/fig-1.png)<!-- -->


```r
#, figures-side, fig.show="hold", out.width="50%"}
plot(posterior_pfizer[,1],trace=TRUE,col="navy",main =  ~ epsilon ~ "Pfizer's vaccine",cex.lab=1.5, cex.axis=1.5, cex.main=1.6 )
```

![](R_project_files/figure-html/figure-1.png)<!-- -->


```r
plot(posterior_moderna[,2],trace=TRUE,col="sienna1",main = "nV.A Moderna's vaccine",cex.lab=1.5, cex.axis=1.5, cex.main=1.6 )
```

![](R_project_files/figure-html/fig2-1.png)<!-- -->



```r
plot(posterior_pfizer[,2],trace=TRUE,col="sienna1",main ="nV.A Pfizer's vaccine",cex.lab=1.5, cex.axis=1.5, cex.main=1.6 )
```

![](R_project_files/figure-html/figura2-1.png)<!-- -->




```r
# Calculation of the MODE 

# Moderna
samples_matrixM <- as.matrix(posterior_moderna, chains = TRUE) # this method removes the MCMC attributes from an mcmc object and returns a matrix
head(samples_matrixM,5)
```

```
##      CHAIN   epsilon nV.A
## [1,]     1 0.8479389   87
## [2,]     1 0.9769443   93
## [3,]     1 0.9494604   99
## [4,]     1 0.9520230  104
## [5,]     1 0.9368510   99
```

```r
density_M <- density(samples_matrixM[,2]) # compute the the density with "density" function
#plot(density_M) --> it is the same as the previous plot
index_maxM <- which.max(density_M$y)

# Pfizer
samples_matrixP <- as.matrix(posterior_pfizer, chains = TRUE)
density_P <- density(samples_matrixP[,2])
#plot(density_P)  --> it is the same as the previous plot
index_maxP <- which.max(density_P$y)


modeM <- density_M$x[index_maxM]
cat(paste("The MODE of the density of Moderna's vaccine efficacy is", modeM ,"\n"))
```

```
## The MODE of the density of Moderna's vaccine efficacy is 0.946254652781702
```

```r
modeP <- density_P$x[index_maxP]
cat(paste("The MODE of the density of Pfizer's vaccine efficacy is", modeP ,"\n"))
```

```
## The MODE of the density of Pfizer's vaccine efficacy is 0.952883015382407
```


```r
Summary_Moderna$quantiles
```

```
##               2.5%        25%        50%       75%       97.5%
## epsilon  0.8658914  0.9164509  0.9368343  0.953451   0.9758925
## nV.A    64.0000000 80.0000000 89.0000000 98.000000 117.0000000
```

```r
cat(paste('The 95% credible interval for Moderna efficacy with DIRECT method is [',round(Summary_Moderna$quantiles[1,1]*100,1),'%,',round(Summary_Moderna$quantiles[1,5]*100,1),'%] \n '))
```

```
## The 95% credible interval for Moderna efficacy with DIRECT method is [ 86.6 %, 97.6 %] 
## 
```


```r
CI <- function(x,posterior,delta){
  l <- x[which.min(abs(cumsum(posterior*delta)-0.025))]
  h <- x[which.min(abs(cumsum(posterior*delta)-0.975))]
  newList <- list("low" = l, "high" = h)
  return(newList)
} 

#function parameters for Moderna
xM <- density_M$x
posteriorM <- density_M$y
deltaM <- xM[2]-xM[1]

#function parameters for Pfizer
xP <- density_P$x
posteriorP <- density_P$y
deltaP <- xP[2]-xP[1]

lowM <- CI(xM,posteriorM,deltaM)$low 
highM <- CI(xM,posteriorM,deltaM)$high

lowP <- CI(xP,posteriorP,deltaP)$low
highP <- CI(xP,posteriorP,deltaP)$high

cat(paste('The 95% credible interval for Moderna efficacy is [',lowM,',',highM,'] \n '))
```

```
## The 95% credible interval for Moderna efficacy is [ 0.86540795719223 , 0.97571042559337 ] 
## 
```




```r
cat(paste('The 95% credible interval for Pfizer efficacy is [',lowP,',',highP,']'))
```

```
## The 95% credible interval for Pfizer efficacy is [ 0.900493732471465 , 0.974864532687698 ]
```


```r
Summary_Pfizer$quantiles
```

```
##                2.5%         25%        50%         75%       97.5%
## epsilon   0.9008225   0.9329089   0.946447   0.9579148   0.9750515
## nV.A    128.0000000 149.0000000 161.000000 173.0000000 198.0000000
```

```r
cat(paste('The 95% credible interval for Pfizer efficacy with DIRECT method is [',round(Summary_Pfizer$quantiles[1,1]*100,digit=1),'% ,',round(Summary_Pfizer$quantiles[1,5]*100,digits=1),'%] \n '))
```

```
## The 95% credible interval for Pfizer efficacy with DIRECT method is [ 90.1 % , 97.5 %] 
## 
```



```r
# MODERNA plot
plot(posterior_moderna[,1],trace=FALSE,col="navy",xlab=expression(epsilon),main="Density of" ~ epsilon ~ "Moderna's vaccine",lwd=2,cex.lab=1.5, cex.axis=1.5, cex.main=1.6)
abline(v = epsilonM_mean, col="red", lwd=3, lty=2)
abline(v = density_M$x[index_maxM], col="navy", lwd=3, lty=2)
text(density_M$x[index_maxM],         # position of text on x-axis
     5,                                               # position of text on y-axis
     round(density_M$x[index_maxM],digits = 3),         # text to be plotted
     pos = 4,col = "navy",cex=1.5) 
text(0.90,                                    # position of text on x-axis
     6,                                               # position of text on y-axis
     round(epsilonM_mean,3),         # text to be plotted
     pos = 4,col = "red",cex=1.5) 
rect(lowM, 0, highM, 16, density = 10, border = "azure3", col = "azure3")
       
legend(0.76, 10, legend=c("mean", "mode","95% CI"),
       col=c("red", "navy","azure3"), lty=c(2,2,2),lwd=c(3,3,10), cex=1.3) 
#par(new=TRUE)

# PFIZER plot
plot(posterior_pfizer[,1],trace=FALSE,col="navy",xlab=expression(epsilon),main="Density of" ~ epsilon ~ "Pfizer's vaccine",lwd=2,cex.lab=1.5, cex.axis=1.5,cex.main=1.6)
abline(v = epsilonP_mean, col="red", lwd=3, lty=2)
abline(v = density_P$x[index_maxP], col="navy", lwd=3, lty=2)
text(density_P$x[index_maxP],         # position of text on x-axis
     10,                                               # position of text on y-axis
     round(density_P$x[index_maxP],digits = 3),         # text to be plotted
     pos = 4,col = "navy",cex=1.5) 
text(0.92,         # position of text on x-axis
     8,                                               # position of text on y-axis
     round(epsilonP_mean,3),         # text to be plotted
     pos = 4,col = "red",cex=1.5) 

rect(lowP, 0, highP, 25,density=10, border = "azure3", col = "azure3")
legend(0.82, 13, legend=c("mean", "mode","95% CI"),
       col=c("red", "navy","azure3"), lty=c(2,2,2),lwd=c(3,3,10), cex=1.3) 
```

<img src="R_project_files/figure-html/plottini, figures-side-1.png" width="50%" /><img src="R_project_files/figure-html/plottini, figures-side-2.png" width="50%" />

The *media claims* and the *press release* results of the two companies correspond to the *modal values* of the two distribution


```r
cat(paste("The mean and the corresponding std for Moderna's efficacy \n",round(epsilonM_mean,3) ,"+/-",round(epsilonM_std,3),"\n "))
```

```
## The mean and the corresponding std for Moderna's efficacy 
##  0.933 +/- 0.029 
## 
```


```r
cat(paste("The mean and the corresponding std for Pfizer's efficacy \n",round(epsilonP_mean,3) ,"+/-",round(epsilonP_std,3),"\n "))
```

```
## The mean and the corresponding std for Pfizer's efficacy 
##  0.944 +/- 0.019 
## 
```



```r
cat(paste("The mean and the corresponding std for Moderna's nV.A \n",round(mean.nV.A_M,0) ,"+/-",round(std.nV.A_M,0),"\n" ))
```

```
## The mean and the corresponding std for Moderna's nV.A 
##  89 +/- 13
```


```r
cat(paste("The mean and the corresponding std for Pfizer's nV.A \n",round(mean.nV.A_P,0),"+/-",round(std.nV.A_P,0),"\n" ))
```

```
## The mean and the corresponding std for Pfizer's nV.A 
##  161 +/- 18
```

Same order of magnitude of the number of infected of the placebo groups, $n_{P_{I}}$ 
<br />
<br />
<center>
Moderna: $n_{P_{I}}$=90

Pfizer: $n_{P_{I}}$=162
</center>
# Histrograms of the MCMC data

```r
posterior_P.df <- as.data.frame(as.mcmc(posterior_pfizer))
head(posterior_P.df,5)
```

```
##     epsilon nV.A
## 1 0.9514116  163
## 2 0.9583534  179
## 3 0.9295840  162
## 4 0.9546733  158
## 5 0.9249872  142
```


```r
posterior_M.df <- as.data.frame(as.mcmc(posterior_moderna))
head(posterior_M.df,5)
```

```
##     epsilon nV.A
## 1 0.8479389   87
## 2 0.9769443   93
## 3 0.9494604   99
## 4 0.9520230  104
## 5 0.9368510   99
```


```r
hist(posterior_P.df$epsilon,density=40, nc=50, prob=TRUE , col='navy',
xlab=expression(epsilon), ylab='f(p)', main='Inference on'~ epsilon ~ "Pfizer" , cex.lab=1.5, cex.axis=1.5,cex.main=1.6)
abline(v = mean(posterior_P.df$epsilon), col="black", lwd=3, lty=2)
text(mean(posterior_P.df$epsilon),         # position of text on x-axis
     10,                 # position of text on y-axis
     round(mean(posterior_P.df$epsilon),3),         # text to be plotted
     pos = 4,col = "black",cex=1.5) 
par(new=TRUE)
plot(posterior_pfizer[,1],trace=FALSE,col="navy",axes=FALSE,lwd=3, lty=2)

legend(0.83, 15, legend=c("mean","jags result"),
       col=c("black","navy"), lty=c(2,2),lwd=c(3,3), cex=1.3) 


plot(posterior_pfizer[,2],trace=FALSE,col='#FF000088',lwd=3, lty=2)
par(new=TRUE)
hist(posterior_P.df$nV.A, density=40,nc=50, prob=TRUE ,axes=FALSE, col='blue',
xlab='nV.A', ylab='f(nV.A)', main='Inference on nV.A, Pfizer', cex.lab=1.5, cex.axis=1.5,cex.main=1.6)
abline(v = mean.nV.A_P, col="black", lwd=3, lty=2)
text(mean.nV.A_P,         # position of text on x-axis
     0.010,                 # position of text on y-axis
     round(mean.nV.A_P,0),         # text to be plotted 
     pos = 4,col = "black",cex=1.5) 


legend(203, 0.015, legend=c("mean","jags result"),
       col=c("black",'#FF000088'), lty=c(2,1),lwd=c(3,10), cex=1.3) 
```

<img src="R_project_files/figure-html/figures-side-1.png" width="50%" /><img src="R_project_files/figure-html/figures-side-2.png" width="50%" />

```r
hist(posterior_M.df$epsilon,density=40, nc=50, prob=TRUE , col='navy',
xlab=expression(epsilon), ylab='f(p)', main='Inference on'~ epsilon ~ ", Moderna" , cex.lab=1.5, cex.axis=1.5,cex.main=1.6)
abline(v = mean(posterior_M.df$epsilon), col="black", lwd=3, lty=2)
text(0.93,         # position of text on x-axis
     5,                 # position of text on y-axis
     round(mean(posterior_M.df$epsilon),3),         # text to be plotted
     pos = 4,col = "black",cex=1.5) 

legend(0.76, 12, legend=c("mean","jags result"),
       col=c("black","navy"), lty=c(2,2),lwd=c(3,3), cex=1.3) 
par(new=TRUE)
plot(posterior_moderna[,1],trace=FALSE, axes=FALSE,col="navy",lty=2,lwd=3)

plot(posterior_moderna[,2],trace=FALSE,col='#FF000088',lwd=3, lty=2)
par(new=TRUE)
hist(posterior_M.df$nV.A, density=40,nc=50, prob=TRUE ,axes=FALSE, col='blue',
xlab='nV.A', ylab='f(nV.A)', main='Inference on nV.A, Moderna', cex.lab=1.5, cex.axis=1.5,cex.main=1.6)
abline(v = mean.nV.A_M, col="black", lwd=3, lty=2)
text(mean.nV.A_M,         # position of text on x-axis
     0.010,                 # position of text on y-axis
     round(mean.nV.A_M,0),         # text to be plotted
     pos = 4,col = "black",cex=1.5) 

legend(120, 0.025, legend=c("mean","jags result"),
       col=c("black","#FF000088"), lty=c(2,1),lwd=c(3,10), cex=1.3) 
```

<img src="R_project_files/figure-html/hh, figures-side-1.png" width="50%" /><img src="R_project_files/figure-html/hh, figures-side-2.png" width="50%" />










# Convergence diagnostic 
If the run of the Markov Chain is long enough a random draw from the chain can be considered as a random draw from the posterior. Problem: one may get a chain not representative *at all* for the posterior. 
<br />
<br />
The trace plot provides "informal" diagnostic for the convergence of our chain.
<br />
Fortunately the `coda` package provides many convergence diagnostics. 


## Geweke diagnostic
<br />
<br />
$X_{1}$ and $X_{2}$ are the two samples, $T$ mean test
<br />
<br />
<center>
$T= \frac{\bar{X_{1}}-\bar{X_{2}}}{\sqrt{\frac{s1^2}{n}+\frac{s2^2}{m}}}$
</center>
<br />
When $n,m \rightarrow \infty$ $T$ can be approximated using the standard normal $Z$.
<br />
<br />
<center>
$Z = \frac{X-\mu}{\sigma}$
</center>
<br />

```r
cat(paste("Z-score of Moderna efficacy chain"),"\n")
```

```
## Z-score of Moderna efficacy chain
```

```r
geweke.diag(posterior_moderna[,1])     # z scores for a test of quality
```

```
## [[1]]
## 
## Fraction in 1st window = 0.1
## Fraction in 2nd window = 0.5 
## 
##   var1 
## 0.7973
```

```r
cat(paste("Z-score of Moderna nV_A chain"),"\n")
```

```
## Z-score of Moderna nV_A chain
```

```r
geweke.diag(posterior_moderna[,2])
```

```
## [[1]]
## 
## Fraction in 1st window = 0.1
## Fraction in 2nd window = 0.5 
## 
##    var1 
## -0.2989
```

```r
cat(paste("Z-score of Pfizer efficacy chain"),"\n")
```

```
## Z-score of Pfizer efficacy chain
```

```r
geweke.diag(posterior_pfizer[,1])
```

```
## [[1]]
## 
## Fraction in 1st window = 0.1
## Fraction in 2nd window = 0.5 
## 
##   var1 
## 0.1692
```

```r
cat(paste("Z-score of Pfizer nV_A chain"),"\n")
```

```
## Z-score of Pfizer nV_A chain
```

```r
geweke.diag(posterior_pfizer[,2])
```

```
## [[1]]
## 
## Fraction in 1st window = 0.1
## Fraction in 2nd window = 0.5 
## 
##   var1 
## 0.4887
```


## Autocorrelation plots
In a MCMC the samples will typically be autocorrelated (or anticorrelated) within a chain.

```r
autocorr.plot(posterior_moderna[,1],main="Moderna"~ epsilon, cex.lab=1.5, cex.axis=1.5,cex.main=1.6,col="navy")
text(40,0.80,sprintf("effective size: %.2f",effectiveSize(posterior_moderna[,1])),cex=1)
autocorr.plot(posterior_moderna[,2],main="Moderna nV.A", cex.lab=1.5, cex.axis=1.5,cex.main=1.3,col="sienna1")
text(40,0.80,sprintf("effective size: %.2f",effectiveSize(posterior_moderna[,2])),cex=1)
```

<img src="R_project_files/figure-html/autocorr, figures-side-1.png" width="50%" /><img src="R_project_files/figure-html/autocorr, figures-side-2.png" width="50%" />

```r
autocorr.plot(posterior_pfizer[,1],main="Pfizer"~ epsilon, cex.lab=1.5, cex.axis=1.5,cex.main=1.6,col="navy")
text(40,0.80,sprintf("effective size: %.2f",effectiveSize(posterior_pfizer[,1])),cex=1)
autocorr.plot(posterior_pfizer[,2],main="Pfizer nV.A", cex.lab=1.5, cex.axis=1.5,cex.main=1.3,col="sienna1")
text(40,0.80,sprintf("effective size: %.2f",effectiveSize(posterior_pfizer[,2])),cex=1)
```

<img src="R_project_files/figure-html/autocorrP, figures-side-1.png" width="50%" /><img src="R_project_files/figure-html/autocorrP, figures-side-2.png" width="50%" />




# Size-independent model
Proof that the result for the *efficacy* does not depend on the initial $n_{V}$ and $n_{P}$. The proof is only for Moderna’s vaccine. 

```r
model_jags_proof <- function(nP) {
  M <- jags.model(textConnection(model_vaccine),data=list(nV.I=5,nP.I=90,nV=nP,nP=nP),inits = list(.RNG.name= "base::Wichmann-Hill", .RNG.seed=1234))
  return(M) 
}
n_init <- list(300,1500,15000)
moderna0 <- model_jags_proof(n_init[1])
```

```
## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 2
##    Unobserved stochastic nodes: 3
##    Total graph size: 9
## 
## Initializing model
```

```r
moderna1 <- model_jags_proof(n_init[2])
```

```
## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 2
##    Unobserved stochastic nodes: 3
##    Total graph size: 9
## 
## Initializing model
```

```r
moderna2 <- model_jags_proof(n_init[3])
```

```
## Compiling model graph
##    Resolving undeclared variables
##    Allocating nodes
## Graph information:
##    Observed stochastic nodes: 2
##    Unobserved stochastic nodes: 3
##    Total graph size: 9
## 
## Initializing model
```


```r
update(moderna0,1000)
update(moderna1,1000)
update(moderna2,1000)
```


```r
posterior_moderna0 <- draw.posterior(moderna0)
posterior_moderna1 <- draw.posterior(moderna1)
posterior_moderna2 <- draw.posterior(moderna2)
```


```r
plot(posterior_moderna0[,1],trace=FALSE,col="navy",main="Density of" ~ epsilon ~ "Moderna's vaccine",xlab=expression(epsilon),lwd=2,lty=2)
par(new=TRUE)
plot(posterior_moderna1[,1],trace=FALSE,col="firebrick",xlab=expression(epsilon),lwd=2,lty=2,axes=FALSE)
par(new=TRUE)
plot(posterior_moderna2[,1],trace=FALSE,col="green",xlab=expression(epsilon),lwd=2,lty=2,axes=FALSE)

legend(0.78, 13, legend=c("nV=nP=300", "nV=nP=1500","nV=nP=15000"),
       col=c("navy", "firebrick","green"), lty=c(2,2,2),lwd=c(2,2,2), cex=1.1) 
```

![](R_project_files/figure-html/unnamed-chunk-31-1.png)<!-- -->



# Conclusion
- The performances of the two vaccines in terms of efficacy do not differ much

```r
cat(paste("The mean and the corresponding std for Moderna's efficacy \n (",round(epsilonM_mean*100,digits=1) ,"+/-",round(epsilonM_std*100,digits=1),") % \n"))
```

```
## The mean and the corresponding std for Moderna's efficacy 
##  ( 93.3 +/- 2.9 ) %
```


```r
cat(paste("The mean and the corresponding std for Pfizer's efficacy \n (",round(epsilonP_mean*100,digits=1) ,"+/-",round(epsilonP_std*100,digits=1),") %"))
```

```
## The mean and the corresponding std for Pfizer's efficacy 
##  ( 94.4 +/- 1.9 ) %
```

```r
cat(paste('The MODE and the 95% CI for Moderna efficacy is \n',round(modeM*100,1),'%,  [',round(Summary_Moderna$quantiles[1,1]*100,1),'%,',round(Summary_Moderna$quantiles[1,5]*100,1),'%] \n '))
```

```
## The MODE and the 95% CI for Moderna efficacy is 
##  94.6 %,  [ 86.6 %, 97.6 %] 
## 
```

```r
cat(paste('The MODE and the 95% CI for Pfizer efficacy is \n',round(modeP*100,1),'%, [',round(Summary_Pfizer$quantiles[1,1]*100,1),'%,',round(Summary_Pfizer$quantiles[1,5]*100,1),'%] \n '))
```

```
## The MODE and the 95% CI for Pfizer efficacy is 
##  95.3 %, [ 90.1 %, 97.5 %] 
## 
```
*Bare* claims: 94.5% for Moderna and 95% for Pfizer

- Simplifying assumptions may affect the result (pA)
- The model allows us to infere the quantities of interest from few data publicly available
- The uncertainity over the efficacy must be present

# Reference
G. D'Agostini, A. Esposito **Inferring vaccine efficacies and their uncertainty** (2020)





