"""Support ticket triage: TF-IDF baseline vs DistilBERT full fine-tuning vs LoRA."""


def main() -> None:
    from support_ticket_triage import baseline

    baseline.main()
