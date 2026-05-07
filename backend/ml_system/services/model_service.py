import mlflow.pyfunc

MODEL_NAME = "bank_marketing_model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

model = mlflow.pyfunc.load_model(MODEL_URI)

train_data = pd.read_csv("data/processed/X_train.csv")