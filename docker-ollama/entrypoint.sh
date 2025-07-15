#!/bin/bash

echo "🚀 Starting Ollama..."

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

# Wait for ollama to start up
echo "⏳ Wait Ollama to be ready..."
sleep 15

# Check Ollama is ready
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "   Ollama non ancora pronto, aspetto..."
    sleep 3
done

echo "✅ Ollama ready"

# download models if not already presents
echo "📦 Checking models..."

EXISTING_MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d':' -f2 | tr -d '"')

# models to download
MODELS=(
    "tinyllama:1.1b"
    "llama3.1:8b"
    "mistral:7b"
)

for model in "${MODELS[@]}"; do
    if echo "$EXISTING_MODELS" | grep -q "$model"; then
        echo "✅ $model already installed"
    else
        echo "📦 Pull $model..."
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "✅ $model downloaded"
        else
            echo "❌ error downloading $model"
        fi
    fi
done

echo "🎉 Entrypoint completed"
echo "📋 models installed:"
ollama list

echo "🔄 Ollama is ready"
wait $OLLAMA_PID