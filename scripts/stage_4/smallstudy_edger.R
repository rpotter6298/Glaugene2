# Stage 4 — 5-vs-5 resampling through edgeR-QL; called by fig_smallstudy_noise.py.
# The pydeseq2 path is not the test that produces the reported DE, so the same draws
# (same seeds as the pydeseq2 path) run here too; the comparison then carries no
# engine difference. Held fixed to match the pydeseq2 path: full-cohort filter (a
# real 5-per-group study would filter on its own 10 samples), CPM within the draw,
# design ~ disease only (single-sex draws make ~ sex + disease rank-deficient).
suppressMessages(library(edgeR))

args     <- commandArgs(trailingOnly = TRUE)
counts_f <- args[1]; meta_f <- args[2]; draws_f <- args[3]
hits_f   <- args[4]; summ_f <- args[5]; pthr <- as.numeric(args[6])

counts <- as.matrix(read.delim(counts_f, row.names = 1, check.names = FALSE))
meta   <- read.delim(meta_f, row.names = 1, check.names = FALSE)
draws  <- read.delim(draws_f, check.names = FALSE, colClasses = "character")
# draws: column 1 = seed, columns 2..11 = sample names, cases first then controls
seeds  <- draws[[1]]
cols   <- as.matrix(draws[, -1, drop = FALSE])

one <- function(take) {
  sub <- counts[, take, drop = FALSE]
  md  <- meta[take, , drop = FALSE]
  md$disease <- factor(md$disease, levels = c("control", "diseased"))
  if (nlevels(droplevels(md$disease)) < 2L) return(NULL)
  y <- DGEList(counts = sub)
  # edgeR scales by lib.size * norm.factors, so the draw's own column sums with norm.factors 1 is CPM
  y$samples$lib.size     <- colSums(sub)
  y$samples$norm.factors <- 1
  design <- model.matrix(~ disease, data = md)
  yy  <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(yy, design, robust = TRUE)
  tt  <- topTags(glmQLFTest(fit, coef = 2), n = Inf, sort.by = "none")$table
  tt[!is.na(tt$PValue), c("logFC", "PValue")]
}

hits <- list(); summ <- list()
for (i in seq_along(seeds)) {
  tt <- try(one(cols[i, ]), silent = TRUE)
  if (inherits(tt, "try-error") || is.null(tt)) {
    summ[[i]] <- data.frame(seed = seeds[i], tested = NA_integer_,
                            n_p10 = NA_integer_, n_p05 = NA_integer_)
    next
  }
  h <- tt[tt$PValue < pthr, , drop = FALSE]
  if (nrow(h)) {
    hits[[length(hits) + 1L]] <- data.frame(seed = seeds[i], entry = rownames(h),
                                            logFC = h$logFC, PValue = h$PValue)
  }
  summ[[i]] <- data.frame(seed = seeds[i], tested = nrow(tt),
                          n_p10 = nrow(h), n_p05 = sum(tt$PValue < 0.05))
  if (i %% 50 == 0) cat(sprintf("  %d/%d draws\n", i, length(seeds)))
}
write.table(do.call(rbind, summ), summ_f, sep = "\t", quote = FALSE, row.names = FALSE)
write.table(do.call(rbind, hits), hits_f, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("  wrote %d draws\n", length(seeds)))
