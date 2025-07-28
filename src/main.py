from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # 1. Ingest data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Transform data
    transformer = DataTransformation()
    train_array, test_array, _ = transformer.initiate_data_transformation(train_path, test_path)

    # 3. Train model
    trainer = ModelTrainer()
    score = trainer.initiate_model_trainer(train_array, test_array)

    print(f"\n✅ Model trained successfully. R2 Score: {score:.4f}")
