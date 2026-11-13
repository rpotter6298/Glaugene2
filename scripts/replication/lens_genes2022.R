# Replication B — Genes 2022 lens capsule, their pipeline exactly.
#   Tomczyk-Socha et al., Genes 2022, 13(4), 582
# Called by lens_genes2022.py, never run directly.
# Their analysis layer (their section 2.2): TMM via calcNormFactors (not injected
# size factors), the 3.7 CPM filter applied by the caller, edgeR NB likelihood
# ratio test at p < 0.1 (their threshold), disease-only design.
suppressMessages(library(edgeR))

args <- commandArgs(trailingOnly = TRUE)
counts_f <- args[1]; meta_f <- args[2]; out_f <- args[3]

counts <- as.matrix(read.delim(counts_f, row.names = 1, check.names = FALSE))
meta   <- read.delim(meta_f, row.names = 1, check.names = FALSE)
meta   <- meta[colnames(counts), , drop = FALSE]

y <- DGEList(counts = counts, group = meta$disease)
y <- calcNormFactors(y, method = "TMM")
design <- model.matrix(~ disease, data = meta)
y <- estimateDisp(y, design, robust = TRUE)

fit <- glmFit(y, design)
lrt <- glmLRT(fit, coef = 2)
res <- topTags(lrt, n = Inf, sort.by = "none")$table
res$entry <- rownames(res)
write.table(res[, c("entry", "logFC", "logCPM", "LR", "PValue", "FDR")],
            out_f, sep = "\t", quote = FALSE, row.names = FALSE)
cat("edgeR LRT done:", nrow(res), "features\n")
