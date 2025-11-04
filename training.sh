#!/bin/bash

# Mac-specific environment setup
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "🚀 Starting Retail AI Assistant Training Pipeline"

# Activate virtual environment
source venv/bin/activate

# Step 1: Generate dataset
echo "📊 Generating training data..."
python data/create_retail_dataset.py

# Step 2: Fine-tune model
echo "🎯 Fine-tuning model..."
python scripts/finetune_model.py

# Step 3: Test inference
echo "✅ Testing model inference..."
python -c "
from scripts.inference import RetailAssistant
assistant = RetailAssistant()
response = assistant.generate_response(
    'You are a Macy\'s assistant',
    'What black dresses do you have?'
)
print('Test response:', response)
"

echo "✨ Training complete!"