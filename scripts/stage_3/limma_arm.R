# Stage 3 — limma arm, called by de_transform.py: voom or a caller-transformed
# (Box-Cox) matrix, both ~sex+disease with eBayes moderation.
suppressMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]                     # "voom" | "matrix"
in_f <- args[2]; meta_f <- args[3]; sf_f <- args[4]; out_f <- args[5]

X    <- as.matrix(read.delim(in_f, row.names = 1, check.names = FALSE))
meta <- read.delim(meta_f, row.names = 1, check.names = FALSE)[colnames(X), , drop = FALSE]
design <- model.matrix(~ sex + disease, data = meta)

if (mode == "voom") {
  sf <- read.delim(sf_f, row.names = 1, check.names = FALSE)[colnames(X), , drop = FALSE]
  # voom expects library sizes; hand it the caller's size factors on the count scale
  v   <- voom(X, design, lib.size = sf$size_factor * mean(colSums(X)))
  fit <- lmFit(v, design)
} else {
  fit <- lmFit(X, design)           # already normalised + transformed upstream
}
fit <- eBayes(fit, robust = TRUE)
tt  <- topTable(fit, coef = "diseasediseased", number = Inf, sort.by = "none")

out <- data.frame(entry = rownames(tt), log2FC = tt$logFC, p = tt$P.Value,
                  AveExpr = tt$AveExpr, check.names = FALSE)
write.table(out, out_f, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("    %-8s %d entries, p<0.05 %d\n", mode, nrow(out), sum(out$p < 0.05)))
