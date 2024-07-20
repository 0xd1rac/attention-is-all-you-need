from src.imports.common_imports import *
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

class MetricManager:
    @staticmethod
    def get_metric(metric_type, pred_text_lis, tgt_text_lis):
        if metric_type == "cer":
            metric = CharErrorRate()
        elif metric_type == "wer":
            metric = WordErrorRate()
        elif metric_type == "bleu":
            metric = BLEUScore()
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        return metric(pred_text_lis, tgt_text_lis)