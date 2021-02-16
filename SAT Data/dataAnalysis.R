library(Matrix)
library(lme4)
library(lmerTest)
library(boot)

M0 = lmer(RL~ laggedsignalDuration + (1+laggedsignalDuration|rat), data = ratDF3)
summary(M0)
anova(M0)

for (j in 1:5)
{
  curve(coef(M0)$rat[j,1] + coef(M0)$rat[j,2]*x, xlim = c(-0.0,1), ylim = c(-0,0.2),
        add=TRUE)
}

M1 = lmer(RL~signalDuration*miss+hit+cr+incongHit+laggedsignalDuration
          +(1+signalDuration*miss+miss+hit+cr+incongHit+laggedsignalDuration|rat), 
          data = ratDF3)
#laggedRL+signalDuration+hit+miss+cr+incongHit
#laggedRL+signalDuration+response+incongHit+congHit
summary(M1)
Mn = lmer(RL~1 +(1|rat), data = ratDF3)
anova(M1, Mn)

curve(fixef(M1)[1] + fixef(M1)[4]*x, xlim = c(-0.,1), ylim = c(-0,0.2),
      col = 'red')
for (j in 1:54)
{
  curve(coef(M1)$rat[j,1] + coef(M1)$rat[j,4]*x, xlim = c(-0.,1), ylim = c(-0,0.2),
        add=TRUE)
}




#logistic regression for response

Md0 = glmer(response~ laggedsignalDuration + (1+laggedsignalDuration|rat), data = ratDF3, 
            family=binomial)
summary(Md0)
anova(Md0)

for (j in 1:54)
{
  curve(inv.logit(coef(Md0)$rat[j,1]+ coef(Md0)$rat[j,2]*x), xlim = c(-1,2), 
        ylim = c(0,1), add = TRUE)
}

Md1 = glmer(response~laggedResponse+signalDuration+RL+laggedsignalDuration
            +(1+laggedResponse+signalDuration+RL+laggedsignalDuration|rat), 
            data = ratDF3, family = binomial)
summary(Md1)
Mdn = glmer(response~1 + (1|rat), data = ratDF3, family = binomial)
anova(Md1, Mdn)


curve(inv.logit(fixef(Md1)[1]+ fixef(Md1)[3]*x), xlim = c(-2,2), 
      ylim = c(0,1), col = 'red')
for (j in 1:14)
{
  curve(inv.logit(coef(Md1)$rat[j,1]+ coef(Md1)$rat[j,3]*x), xlim = c(-2,2), 
        ylim = c(0,1), add = TRUE)
}
