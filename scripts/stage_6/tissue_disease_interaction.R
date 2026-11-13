# Stage 6 - does the lens-vs-tear difference depend on disease?
# 42 patients gave BOTH tissues (19 control, 23 XFG): the contrast is within
# patient, so between-patient nuisance cancels. Blocking on patient also absorbs
# the disease main effect (constant within a patient); the question is the
# interaction - does the tissue difference itself shift with disease.
#
#   ~ patient + tissue + tissue:disease
# tissue        lens-vs-tear difference in controls
# tissue:disease how that difference changes in XFG   <- the question
suppressMessages(library(edgeR))

args <- commandArgs(trailingOnly = TRUE)
counts <- as.matrix(read.delim(args[1], row.names = 1, check.names = FALSE))
meta   <- read.delim(args[2], row.names = 1, check.names = FALSE)
meta   <- meta[colnames(counts), , drop = FALSE]
meta$patient <- factor(meta$patient)
meta$tissue  <- factor(meta$tissue_name, levels = c("tear", "lens"))
meta$disease <- factor(meta$disease, levels = c("control", "diseased"))

y <- DGEList(counts = counts)
# CPM, matching every other reported analysis (lib.size = own total, norm.factors 1):
# changed from TMM on 2026-10-19 per RP - no conclusion moves either way, and TMM's
# "most features unchanged" assumption is false here, 246 of 290 differing.
y$samples$lib.size     <- colSums(counts)
y$samples$norm.factors <- 1
design <- model.matrix(~ patient + tissue + tissue:disease, data = meta)
design <- design[, qr(design)$pivot[seq_len(qr(design)$rank)], drop = FALSE]
y <- estimateDisp(y, design, robust = TRUE)
fit <- glmQLFit(y, design, robust = TRUE)

cat("  design columns:", paste(tail(colnames(design), 5), collapse=" , "), "\n")
icol <- grep(":disease", colnames(design), fixed = TRUE)
stopifnot(length(icol) == 1)
tt  <- topTags(glmQLFTest(fit, coef = icol), n = Inf, sort.by = "none")$table
tc  <- grep("^tissuelens$", colnames(design))
tt2 <- topTags(glmQLFTest(fit, coef = tc), n = Inf, sort.by = "none")$table

out <- data.frame(entry = rownames(tt),
                  tissue_log2FC = tt2$logFC, tissue_p = tt2$PValue,
                  tissue_FDR = tt2$FDR,
                  inter_log2FC = tt$logFC, inter_p = tt$PValue, inter_FDR = tt$FDR,
                  logCPM = tt$logCPM, check.names = FALSE)
write.table(out, args[3], sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("  %d miRNAs | tissue effect: %d at FDR<0.05 | interaction: %d at FDR<0.05, %d at raw p<0.05\n",
            nrow(out), sum(out$tissue_FDR < 0.05), sum(out$inter_FDR < 0.05),
            sum(out$inter_p < 0.05)))
