get_legend <- function(p){
     tmp <- ggplot_gtable(ggplot_build(p))
     leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
     legend <- tmp$grobs[[leg]]
     
     return(legend)
}