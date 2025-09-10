#!/bin/bash

# AI ATS Python Backend - Quick Restart Script

echo "🔄 Restarting AI ATS Python Backend..."

# Stop the existing process
echo "🛑 Stopping existing PM2 process..."
pm2 stop AI_Ats_python_Backend 2>/dev/null || true
pm2 delete AI_Ats_python_Backend 2>/dev/null || true

# Wait a moment
sleep 2

# Start with the updated configuration
echo "🚀 Starting with updated configuration..."
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Show status
echo "📊 PM2 Status:"
pm2 status AI_Ats_python_Backend

# Show recent logs
echo "📝 Recent logs:"
pm2 logs AI_Ats_python_Backend --lines 20

echo "✅ Restart completed!"
echo "🌐 Test the application: curl http://158.220.127.100:8000/health"
echo "📚 API Docs: https://pyats.workisy.in/docs"
