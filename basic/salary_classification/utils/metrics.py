from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculates and logs evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"--- Evaluation Report for {model_name} ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:\n" + report)
    logger.info("Confusion Matrix:\n" + str(cm))
    logger.info("--- End Report ---")

    return accuracy, report, cm 