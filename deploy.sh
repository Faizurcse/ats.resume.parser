#!/bin/bash

# AI ATS Python Backend - Deployment Script

echo "🚀 Deploying AI ATS Python Backend..."

# Make scripts executable
chmod +x start_production.sh

# Create logs directory
mkdir -p logs

# Install production dependencies
echo "📦 Installing production dependencies..."
pip install -r requirements-prod.txt

# Stop existing PM2 process
echo "🛑 Stopping existing PM2 process..."
pm2 stop AI_Ats_python_Backend 2>/dev/null || true
pm2 delete AI_Ats_python_Backend 2>/dev/null || true

# Start with optimized configuration
echo "🎯 Starting with 6 workers (3 CPU cores × 2)..."
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Show status
echo "📊 PM2 Status:"
pm2 status

echo "✅ Deployment completed successfully!"
echo "🌐 API available at: https://pyats.workisy.in"
echo "📝 View logs: pm2 logs AI_Ats_python_Backend"
echo "📊 Monitor: pm2 monit"
echo "🔄 Restart: pm2 restart AI_Ats_python_Backend"
