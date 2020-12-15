dim(train)
str(train[,c(1:10, 81)])
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL

test$SalePrice <- NA
all <- rbind(train, test)
dim(all)

ggplot(data = all[!is.na(all$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(fill='blue', binwidth = 10000) +
  scale_x_continuous(breaks = seq(0, 8000000,by=100000), labels = comma)

summary(all$SalePrice)
numericVars <- which(sapply(all, is.numeric))
numericVarNames <- names(numericVars)

cat("there are ", length(numericVars), "numeric variables")

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use = 'pairwise.complete.obs')
cor_sorted <- as.matrix(sort(cor_numVar[, 'SalePrice'], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col = 'black', tl.pos = 'lt')
