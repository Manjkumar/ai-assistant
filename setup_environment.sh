#!/bin/bash

# setup_environment.sh - Fixed for Mac

echo "🚀 Setting up Retail AI Assistant Environment for Mac"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Mac
echo "🔥 Installing PyTorch for Mac..."
if [[ $(uname -m) == 'arm64' ]]; then
    # Apple Silicon (M1/M2/M3)
    echo "Detected Apple Silicon Mac"
    pip install torch torchvision torchaudio
else
    # Intel Mac
    echo "Detected Intel Mac"
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install transformers datasets accelerate
pip install pandas numpy scikit-learn
pip install gradio
pip install sentencepiece protobuf

# For Apple Silicon, install specific optimizations
if [[ $(uname -m) == 'arm64' ]]; then
    echo "🍎 Installing Apple Silicon optimizations..."
    pip install tensorflow-macos tensorflow-metal
fi

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"