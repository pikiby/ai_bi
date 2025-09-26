#!/bin/bash

# Простой скрипт для запуска проекта

echo "🚀 Запуск AI BI проекта..."

# Создаем необходимые директории
mkdir -p data/chroma docs

# Запускаем через docker-compose
echo "🐳 Запуск через Docker..."
docker-compose up --build

echo " Приложение будет доступно по адресу: http://localhost:8501"