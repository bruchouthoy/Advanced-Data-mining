########################################
#Brunda Chouthoy
#CSC529 - Case Study 3
# Bayesian Networks - Asia dataset
#########################################

library(bnlearn)

#######################################
#Dataset creation
#######################################
# these are the R commands used to generate this data set.
a = sample(c("yes", "no"), 5000, prob = c(0.01, 0.99), replace = TRUE)
s = sample(c("yes", "no"), 5000, prob = c(0.50, 0.50), replace = TRUE)

t = a
t[t == "yes"] = sample(c("yes", "no"), length(which(t == "yes")),
                       prob = c(0.05, 0.95), replace = TRUE)
t[t == "no"] = sample(c("yes", "no"), length(which(t == "no")),
                      prob = c(0.01, 0.99), replace = TRUE)

l = s
l[l == "yes"] = sample(c("yes", "no"), length(which(l == "yes")),
                       prob = c(0.10, 0.90), replace = TRUE)
l[l == "no"] = sample(c("yes", "no"), length(which(l == "no")),
                      prob = c(0.01, 0.99), replace = TRUE)

b = s
b[b == "yes"] = sample(c("yes", "no"), length(which(b == "yes")),
                       prob = c(0.60, 0.40), replace = TRUE)
b[b == "no"] = sample(c("yes", "no"), length(which(b == "no")),
                      prob = c(0.30, 0.70), replace = TRUE)

e = apply(cbind(l,t), 1, paste, collapse= ":")
e[e == "yes:yes"] = "yes"
e[e == "yes:no"] = "yes"
e[e == "no:yes"] = "yes"
e[e == "no:no"] = "no"

x = e
x[x == "yes"] = sample(c("yes", "no"), length(which(x == "yes")),
                       prob = c(0.98, 0.02), replace = TRUE)
x[x == "no"] = sample(c("yes", "no"), length(which(x == "no")),
                      prob = c(0.05, 0.95), replace = TRUE)

d = apply(cbind(e,b), 1, paste, collapse= ":")
d[d == "yes:yes"] = sample(c("yes", "no"), length(which(d == "yes:yes")),
                           prob = c(0.90, 0.10), replace = TRUE)
d[d == "yes:no"] = sample(c("yes", "no"), length(which(d == "yes:no")),
                          prob = c(0.70, 0.30), replace = TRUE)
d[d == "no:yes"] = sample(c("yes", "no"), length(which(d == "no:yes")),
                          prob = c(0.80, 0.20), replace = TRUE)
d[d == "no:no"] = sample(c("yes", "no"), length(which(d == "no:no")),
                         prob = c(0.10, 0.90), replace = TRUE)

asia <- data.frame(A = a, S = s, T = t, L = l, B = b, E = e, X = x, D = d)
write.csv(asia,file = "asia.csv",row.names = FALSE)
##################################################################################
#Initial data analysis
head(asia)
#check Dimension
dim(asia)
#Check datatype of all features and descriptions of each variable
str(asia)
sapply(asia[1, ],class)
summary(asia)
#Check if there are missing values
table(is.na(asia)) 
sapply(asia, function(asia) sum(is.na(asia)))

##################################################################################

#Network Structure Learning
# using the Grow-Shrink algorithm
bn.gs <- gs(asia)
bn.gs

#using IAMB constraint based algorithms 
bn.iamb<- iamb(asia) #structure learning
bn.iamb
plot(bn.iamb)
bn3 <- fast.iamb(asia)
bn4 <- inter.iamb(asia)

compare(bn.gs, bn.iamb)
compare(bn.gs, bn4)

#score-based learning using the Hill Climbing (HC) algorithm
bn.hc <- hc(asia)
bn.hc
plot(bn.hc)
compare(bn.hc, bn.gs)

#hybrid learning with the Max-Min Hill Climbing (MMHC) algorthim
bn.mmhc <- mmhc(asia)
bn.mmhc
plot(bn.mmhc, main= "Hybrid Learning with MMHC")

#Plots to compare the two network structures 
par(mfrow = c(1,2))
plot(bn.gs, main = "Constraint-based algorithms")
plot(bn.hc, main = "Hill-Climbing")

#Computing the network scores
bn.AB <- gs(asia, blacklist = c("B", "A"))
compare(bn.AB, bn.hc)
score(bn.AB, asia, type = "bde")

bn.BA <- gs(asia, blacklist = c("A", "B"))
compare(bn.BA, bn.hc)
score(bn.BA, asia, type = "bde")

score(bn.hc,asia,type="aic") #getting aic value for full network
score(bn.hc,asia,type="bde")
score(bn.hc,asia,type="bic")

score(bn.mmhc,asia,type="aic") #getting aic value for full network
score(bn.mmhc,asia,type="bde")
score(bn.mmhc,asia,type="bic")

net1 = set.arc(bn.hc, "A", "T") #setting arcs to get actual scores from individual relationships
net2 = set.arc(bn.hc,"T", "A") 
score(net1,asia,type="aic") #retriveing score
score(net2,asia,type="aic") 
score(net2,asia,type="bic")

#Parameter Learning/Training the Network
#creating a new DAG with the A to T relationship
#(based on our previous knowledge that going to asia effects having tuberculosis)
bn.hc = set.arc(bn.hc, from  = "A", to = "T") 
plot(bn.hc, main="Hill Climbing Score based algorithm")

fit = bn.fit(bn.hc, asia) # fitting the network with conditoinal probability tables
fit 
#conditional probability of a specific node
fit$L
fit$D

#visualize the node conditoinal probability tables with barcharts:
bn.fit.barchart(fit$D)
bn.fit.dotplot(fit$D, main = "Conditinal probablities for D -dyspnoea")
bn.fit.dotplot(fit$E, main = "Conditinal probablities for E -tuberculosis versus lung cancer/bronchitis")

#Model Validation - Using 10-fold Cross validation
bn.cv(asia, bn = "hc", runs = 10, algorithm.args = list(score = "bde", iss = 1))
bn.cv(asia, bn = "mmhc", runs = 10, algorithm.args = list(score = "bde", iss = 1))

#Inferences
cpquery(fit, event = (X=="yes"), evidence = ( A=="yes"))
cpquery(fit, event = (A=="yes"), evidence = ( X=="yes"))

cpquery(fit, event = (T=="yes"), evidence = ( A=="yes"))
cpquery(fit, event = (A=="yes"), evidence = ( T=="yes"))

cpquery(fit, event = (S=="yes"), evidence = ( L=="yes"))
cpquery(fit, event = (L=="yes"), evidence = ( S=="yes"))

#Computing scores for the full network
score(bn.hc,asia,type="aic") #retriveing score
score(bn.hc,asia,type="bde") 
score(bn.hc,asia,type="bic")