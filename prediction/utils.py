# from torcheval.metrics.aggregation.auc import AUC
import pandas as pd
from torchmetrics import AUROC
from torchmetrics.classification import MultilabelAUROC
import torch




if __name__ == '__main__':
    auroc = AUROC(task="multilabel", num_labels=8)
    auroc2= MultilabelAUROC(num_labels=8, average='macro', thresholds=None)
    csv_file = 'D:\\ninavv\\phd\\research\\isic_results\\disease\\densenet-tp40-lr1e-05-ep50-pt1-aug1\\predictions.test.csv'
    csv_file = 'D:\\ninavv\\phd\\research\\isic_results\\disease\\densenet-tp40-lr0.0001-ep50-pt1-aug1\\predictions.val.version_2.csv'
    df = pd.read_csv(csv_file)
    preds = torch.tensor(df.iloc[:,:8].values)
    target = torch.tensor(df.iloc[:,16:].values,dtype=torch.int32)
    res = auroc(preds, target)
    res2 =auroc2(preds, target)
    print(res)