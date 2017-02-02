print_result_metrics <- function(predicted, actual) {
     require(Metrics)
     
     error <- percent(Metrics::ce(predicted, actual))
     error_1 <- percent(1- sum(predicted[actual == 1] == actual[actual == 1])/length(actual[actual == 1]))
     
     ## Confusion Table
     names <- list(predicted = c("-$50K", "+$50K"), actual = c("-$50K", "+$50K"))
     
     confusion_table <- table(predicted, actual)
     dimnames(confusion_table) <- names
     
     confusion_table_prct <- array(
          paste(round(100*confusion_table/sum(confusion_table), 2), "%", sep="")
          , dim = c(2, 2)
          , dimnames = names
     )
     
     # Print results
     cat("\n Predicted Error percentage: \n"
         , error
         , "\n"
         , 'Predicted Error percentage of the "positive" elements (people saving more than $50K a year: \n'
         , error_1
         , "\n")
     
     cat("\n Confusion Matrix : \n")
     print(confusion_table)
     cat("\n Confusions Matrix as %: \n")
     print(confusion_table_prct)
}