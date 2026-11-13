# Published human miRNA studies in glaucoma / exfoliation

Assembled 2026-10-07 from PubMed. Columns marked UNVERIFIED could not be settled from
the abstract and need a full-text check before anything is claimed about them in the
manuscript - in particular `correction`, which is the load-bearing column.

## Starting point: the systematic review

Rezaei et al. 2023, Eur J Med Res 28:137, doi:10.1186/s40001-023-01093-8
  - searched PubMed, Web of Science, EMBASE, OVID, Scopus to Dec 2021
  - 648 records -> 321 after deduplication -> 37 full-text -> **6 met inclusion**
  - those 6 compared **115 glaucoma patients with 108 controls IN TOTAL**
  - 52 differentially expressed miRNAs reported across them (28 up, 24 down)
  - only 12 miRNAs had the sensitivity/specificity needed to meta-analyse
  - two tissues each: aqueous humour, tears, serum/plasma
  - the two most frequently reported miRNAs were miR-486-5p and miR-143-3p

FOR SCALE: this study is 94 samples, 49 XFG and 45 cataract-only controls across two
tissues. One study, comparable in size to the entire pooled evidence base of the only
systematic review of the field.

## Two reviews that already concede the problem

Dobrzycka et al. 2023, Int J Mol Sci 24:14699, doi:10.3390/ijms241914699
  "Limitations, such as variability, small sample sizes, sample specificity, and
   limited accessibility to ocular tissues, are also addressed... Reproducibility and
   validation are crucial to establish the credibility of miRNA research findings."

Kapuganti & Alone 2023, Mol Aspects Med 94:101214, doi:10.1016/j.mam.2023.101214
  PEX-specific genetics/epigenetics review; covers the miRNA literature.

## What still has to be checked in full text

For every row marked UNVERIFIED, the manuscript needs: was any multiple-testing
correction applied, what was the tested-feature count, and what threshold was used.
Abstracts routinely omit all three. miR-486-5p and miR-143-3p recur across studies and
both appear in our own raw-p candidate lists - worth checking whether the recurrence
is biology or whether they are simply the ones that surface from noise.

## Excluded, with reason

**Banasaz et al. 2023, Pathol Res Pract 249:154721, doi:10.1016/j.prp.2023.154721**
Removed 2026-10-07 after reading the full text. Not excluded for its findings but
because its reported numbers are not internally consistent, so nothing in it can be
quoted with confidence:

- Sample size given four different ways: abstract 64 PEX + 35 control; Methods 2.3
  30 PEX + 8 NTG + 30 control; Results 3.1 35 PEX + 35 control; Table 1 32 PEX + 35
  control. Table 1's own PEX sex breakdown (13 M + 20 F) totals 33, not 32.
- Age is claimed matched at "P: 0.66". Table 1 gives controls 62.06 +/- 8.37 (n=35)
  and PEX 70.03 +/- 8.99 (n=32): Welch t = 3.75, P = 0.0004, Cohen's d = 0.92. The
  paper's own introduction states miR-34a expression correlates with age, and miR-34a
  is one of its two headline biomarkers - so the unacknowledged 8-year gap is a live
  alternative explanation for the main result.
- Table 1 subgroup means cannot exist: severe PEX 4 M at 72.0 + 3 F at 73.3 gives a
  weighted mean of 72.56, table says 75.57; mild PEX 11 M at 72.0 + 1 F at 63.0 gives
  71.25, table says 63.75.
- Figure 3's caption describes ROC analysis of "miR-28, -31, -34a, and -181a"; miR-28
  and miR-31 were never measured. Table 2's caption lists P53, Rb, TWIST, Bcl-2,
  Bcl-XL, c-IAP2, Cytochrome C and Caspase 3; the table's columns are IL-6, TNF-alpha
  and TGF-beta. Methods 2.6 names the primer "hsa-let-7c-5p"; the study used let-7b.
- Results 3.2 reports miR-29a and miR-126 as showing "a significant elevation" while
  quoting fold changes of -3.38 and -1.83; abstract and Figure 1A both say suppression.
- Methods specify Pearson correlation, Results report Spearman. ROC x-axes are
  labelled "Specificity%" rather than 1 - specificity.

NOTE FOR THE MANUSCRIPT: this is a reporting-integrity problem, which is a DIFFERENT
critique from the statistical one this paper makes (small n plus no correction
manufacturing findings). Do not use it as an example of the latter - conflating the
two weakens the argument and invites a reviewer to check. Record it as excluded and
move on.
