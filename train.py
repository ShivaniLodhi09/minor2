"""Train demand-forecast and anomaly models; save to models/."""
from energy_optimizer.modeling import save_models, train_classifier, train_regressor


def main() -> None:
    print("Training demand forecast (CCPP regression)...")
    reg, reg_metrics = train_regressor()
    print(f"  R² = {reg_metrics['r2']:.4f}, MAE = {reg_metrics['mae']:.2f}")

    print("Training anomaly detector (AI4I classification)...")
    clf, clf_metrics = train_classifier()
    print(f"  Accuracy = {clf_metrics['accuracy']:.4f}, F1 = {clf_metrics['f1']:.4f}")
    print(clf_metrics["report"])

    save_models(reg, clf)
    print("Models saved to models/")


if __name__ == "__main__":
    main()
