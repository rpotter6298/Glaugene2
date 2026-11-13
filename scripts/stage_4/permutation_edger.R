# Stage 4 — permutation null for the edgeR-QL arm; called by permutation_edger.py.
# permutation.py's null is the pydeseq2 Wald arm's, but the p-value histogram and
# volcano are edgeR-QL; this runs the identical shuffles (same seeds as
# permutation.py) through edgeR-QL and returns the full p-value vector of every run.
suppressMessages(library(edgeR))

args     <- commandArgs(trailingOnly = TRUE)
counts_f <- args[1]; meta_f <- args[2]; sf_f <- args[3]
perm_f   <- args[4]; out_f  <- args[5]

counts <- as.matrix(read.delim(counts_f, row.names = 1, check.names = FALSE))
meta   <- read.delim(meta_f, row.names = 1, check.names = FALSE)
sf     <- read.delim(sf_f,   row.names = 1, check.names = FALSE)
perms  <- as.matrix(read.delim(perm_f, row.names = 1, check.names = FALSE))
meta   <- meta[colnames(counts), , drop = FALSE]
sf     <- sf[colnames(counts), , drop = FALSE]
stopifnot(identical(colnames(perms), colnames(counts)))

# size factors are label-independent under CPM: the DGEList is built once, only the design changes
y <- DGEList(counts = counts)
y$samples$lib.size     <- sf$size_factor * mean(colSums(counts))
y$samples$norm.factors <- 1

one <- function(disease) {
  md <- meta
  md$disease <- factor(disease, levels = levels(factor(meta$disease)))
  design <- model.matrix(~ sex + disease, data = md)
  yy  <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(yy, design, robust = TRUE)
  topTags(glmQLFTest(fit, coef = ncol(design)), n = Inf, sort.by = "none")$table$PValue
}

res <- matrix(NA_real_, nrow = nrow(perms) + 1L, ncol = nrow(counts),
              dimnames = list(c("real", rownames(perms)), rownames(counts)))
res[1, ] <- one(meta$disease)
cat(sprintf("  real: QL p<0.05 %d of %d\n", sum(res[1, ] < 0.05), ncol(res)))
for (i in seq_len(nrow(perms))) {
  res[i + 1L, ] <- one(perms[i, ])
  if (i %% 100 == 0) cat(sprintf("  %d/%d shuffles\n", i, nrow(perms)))
}
write.table(res, out_f, sep = "\t", quote = FALSE, col.names = NA)
