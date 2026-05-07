import mlflow

MODEL_NAME = "bank_marketing_model"

def promote_model(model_version):

    client = mlflow.tracking.MlflowClient()

    # move model to Production stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production"
    )

    return True