# Stage 6 - paired lens-vs-tear contrast, fitted separately in each disease group.
# Route A of the paired contrast (route B, the single interaction model, is
# tissue_disease_interaction.R). 42 patients gave BOTH tissues (19 control, 23 XFG),
# so the contrast is WITHIN patient: patient enters the design and between-patient
# nuisance - age, sex, batch, handling - cancels. One fit per disease group:
#
#   ~ patient + tissue
#
# The tissue coefficient is the within-patient lens-vs-tear difference in that
# group. Size factors within each group on the filtered entry set (Section 2.6
# filter on the 84 paired libraries); the same entries are tested in both groups.
suppressMessages(library(edgeR))

args <- commandArgs(trailingOnly = TRUE)
counts <- as.matrix(read.delim(args[1], row.names = 1, check.names = FALSE))
meta   <- read.delim(args[2], row.names = 1, check.names = FALSE)
meta   <- meta[colnames(counts), , drop = FALSE]
meta$patient <- factor(meta$patient)
meta$tissue  <- factor(meta$tissue_name, levels = c("tear", "lens"))

out <- list()
for (d in c('control', 'diseased')) {
  i  <- meta$disease == d
  cc <- counts[, i, drop = FALSE]
  y  <- DGEList(counts = cc)
  # CPM within the group, matching every other reported analysis. Changed from TMM on
  # 2026-10-19 per RP; see tissue_disease_interaction.R for the test that licensed it.
  y$samples$lib.size     <- colSums(cc)
  y$samples$norm.factors <- 1
  ds <- model.matrix(~ patient + tissue, data = meta[i, , drop = FALSE])
  ds <- ds[, qr(ds)$pivot[seq_len(qr(ds)$rank)], drop = FALSE]
  y  <- estimateDisp(y, ds, robust = TRUE)
  f  <- glmQLFit(y, ds, robust = TRUE)
  tc <- grep("^tissuelens$", colnames(ds))
  t  <- topTags(glmQLFTest(f, coef = tc), n = Inf, sort.by = "none")$table
  out[[d]] <- data.frame(entry = rownames(t), lfc = t$logFC, fdr = t$FDR,
                         check.names = FALSE)
}
r <- merge(out[['control']], out[['diseased']], by = 'entry')
colnames(r) <- c('entry', 'lfc_ctrl', 'fdr_ctrl', 'lfc_xfg', 'fdr_xfg')
r <- r[order(r$entry), ]
write.table(r, args[3], sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("  %d miRNAs | tissue contrast: %d at FDR<0.05 in controls, %d in XFG | r(lfc_ctrl, lfc_xfg) = %.4f\n",
            nrow(r), sum(r$fdr_ctrl < 0.05), sum(r$fdr_xfg < 0.05),
            cor(r$lfc_ctrl, r$lfc_xfg)))
