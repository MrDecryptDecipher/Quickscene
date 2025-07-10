#!/bin/bash

# Quickscene Setup Script
# Automated environment setup for development and production

set -e  # Exit on any error

echo "🚀 Setting up Quickscene Video Timestamp-Retrieval System..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{videos,transcripts,chunks,embeddings,index}
mkdir -p logs
echo "✅ Data directories created"

# Download Whisper model (optional, will download on first use)
echo "🤖 Pre-downloading Whisper model..."
python3 -c "import whisper; whisper.load_model('base')" || echo "⚠️  Whisper model will download on first use"

# Verify installation
echo "🧪 Verifying installation..."
python3 -c "
import sys
import whisper
import sentence_transformers
import faiss
import yaml
import numpy as np
print('✅ All core dependencies verified')
"

# Run basic tests if available
if [ -f "tests/test_basic.py" ]; then
    echo "🧪 Running basic tests..."
    python3 -m pytest tests/test_basic.py -v
fi

echo ""
echo "🎉 Quickscene setup complete!"
echo ""
echo "📖 Next steps:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Add videos to: ./data/videos/"
echo "   3. Run processing: python main.py process"
echo "   4. Test queries: python main.py query 'your search query'"
echo ""
echo "📚 Documentation: See README.md for detailed usage"
