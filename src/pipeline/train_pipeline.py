from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Data Ingestion
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Data Transformation
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

# Model Training
model_trainer = ModelTrainer()
print(model_trainer.initiate_model_training(train_arr, test_arr))
