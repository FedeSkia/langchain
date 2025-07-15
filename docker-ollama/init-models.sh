#!/bin/bash

echo "🚀 Avviando Ollama..."

# Avvia Ollama in background
ollama serve &
OLLAMA_PID=$!

# Aspetta che Ollama sia pronto
echo "⏳ Aspettando che Ollama sia pronto..."
sleep 15

# Controlla se Ollama è accessibile
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "   Ollama non ancora pronto, aspetto..."
    sleep 3
done

echo "✅ Ollama è pronto!"

# Scarica i modelli se non esistono
echo "📦 Verificando modelli..."

# Controlla se i modelli esistono già
EXISTING_MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d':' -f2 | tr -d '"')

# Lista dei modelli da scaricare
MODELS=(
    "tinyllama:1.1b"
    "llama3.1:8b"
    "mistral:7b"
)

for model in "${MODELS[@]}"; do
    if echo "$EXISTING_MODELS" | grep -q "$model"; then
        echo "✅ $model già presente"
    else
        echo "📦 Scaricando $model..."
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo "✅ $model scaricato con successo!"
        else
            echo "❌ Errore nel download di $model"
        fi
    fi
done

echo "🎉 Inizializzazione completata!"
echo "📋 Modelli installati:"
ollama list

# Mantiene Ollama in esecuzione
echo "🔄 Ollama è pronto per LangChain!"
wait $OLLAMA_PID