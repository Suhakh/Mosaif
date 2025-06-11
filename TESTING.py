
from final_prediction import GeneticInput, predict_genetic_prob

sample_genetic_input = GeneticInput(
    CHR_ID=1,
    CHR_POS=123456.0,
    SNPS=2,
    SNP_ID_CURRENT=1.0,
    INTERGENIC=0.0,
    RISK_ALLELE_FREQUENCY=0.3,
    P_VALUE=0.001,
    PVALUE_MLOG=3.0,
    OR_or_BETA=1.2,
    PRS_scaled=2.5,
    Ethnicity_African=0,
    Ethnicity_Asian=0,
    Ethnicity_European=1
)

print(predict_genetic_prob(sample_genetic_input))
