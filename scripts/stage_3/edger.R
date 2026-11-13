# Stage 3 — edgeR arm, called by de_edger.py: LRT (matches Genes 2022) + QL F-test.
# Caller's size factors become effective library sizes, so both norm families run
# through identical dispersion estimation and testing.
suppressMessages(library(edgeR))

args    <- commandArgs(trailingOnly = TRUE)
counts_f <- args[1]; meta_f <- args[2]; sf_f <- args[3]; out_f <- args[4]

counts <- as.matrix(read.delim(counts_f, row.names = 1, check.names = FALSE))
meta   <- read.delim(meta_f, row.names = 1, check.names = FALSE)
sf     <- read.delim(sf_f, row.names = 1, check.names = FALSE)
meta   <- meta[colnames(counts), , drop = FALSE]
sf     <- sf[colnames(counts), , drop = FALSE]

y <- DGEList(counts = counts)
# edgeR uses lib.size * norm.factors: set lib.size to the caller's size factor and neutralise norm.factors
y$samples$lib.size     <- sf$size_factor * mean(colSums(counts))
y$samples$norm.factors <- 1

design <- model.matrix(~ sex + disease, data = meta)
y <- estimateDisp(y, design, robust = TRUE)

# LRT — what Genes 2022 used
fit_lrt <- glmFit(y, design)
lrt     <- glmLRT(fit_lrt, coef = "diseasediseased")
r_lrt   <- topTags(lrt, n = Inf, sort.by = "none")$table

# quasi-likelihood F-test — edgeR's current recommendation
fit_ql  <- glmQLFit(y, design, robust = TRUE)
qlf     <- glmQLFTest(fit_ql, coef = "diseasediseased")
r_ql    <- topTags(qlf, n = Inf, sort.by = "none")$table

# 1 numerator df: F = t^2, so the SE is recoverable as |log2FC|/sqrt(F); exported
# with the df so stage_4 equivalence can use the lead method's SE.
out <- data.frame(
  entry       = rownames(r_lrt),
  logCPM      = r_lrt$logCPM,
  lrt_log2FC  = r_lrt$logFC,  lrt_p = r_lrt$PValue,
  ql_log2FC   = r_ql$logFC,   ql_p  = r_ql$PValue,
  ql_F        = r_ql$F,       ql_df = qlf$df.total,
  common_disp = y$common.dispersion,
  check.names = FALSE
)
write.table(out, out_f, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("  edgeR: %d entries, common dispersion %.4f, LRT p<0.05 %d, QL p<0.05 %d\n",
            nrow(out), y$common.dispersion, sum(out$lrt_p < 0.05), sum(out$ql_p < 0.05)))
