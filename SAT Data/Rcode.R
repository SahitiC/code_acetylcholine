library(Matrix)
library(lme4)
library(lmerTest)
library(boot)

M0 = lmer(RL~response + (1+response|rats), data = ratDF)
summary(M0)
anova(M0)

for (j in 1:5)
{
  curve(coef(M0)$rats[j,1] + coef(M0)$rats[j,2]*x,xlim = c(-2,2), ylim = c(-2,2),
        add=TRUE)
}


M1 = lmer(RL~laggedRL+signalDuration+response+incongHit+congHit+time 
            +(1+laggedRL+signalDuration+response+incongHit+congHit+time|rats), 
          data = ratDF)
summary(M1)
Mn = lmer(RL~1 +(1|rats), data = ratDF)
summary(Mn)
anova(M1, Mn)



#logistic regression for response

Md0 = glmer(response~ laggedResponse + (1+laggedResponse|rats), data = ratDF, 
            family=binomial)
summary(Md0)
anova(Md0)

for (j in 1:5)
{
  curve(inv.logit(coef(Md0)$rats[j,1]+ coef(Md0)$rats[j,2]*x), xlim = c(-4,4), 
      ylim = c(0,1), add = TRUE)
}

Md1 = glmer(response~laggedResponse+signalDuration+RL+time 
           +(1+laggedResponse+signalDuration+RL+time|rats), 
           data = ratDF, family = binomial)
summary(Md1)

Mdn = glmer(response~1 + (1|rats), data = ratDF, family = binomial)
summary(Mdn)
anova(Md1, Mdn)