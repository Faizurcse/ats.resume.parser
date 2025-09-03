module.exports = {
  apps: [{
    name: 'AI_Ats_python_Backend',
    script: 'python',
    args: '-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 6',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PORT: '8000',
      OPENAI_API_KEY: 'k-proj-IcwG21HhI9Yiue94-vdiRh_87PrrkJaWqKfM672mtaoWaTZ5tagpisAlQV5bGO46yBXqCSf9UET3BlbkFJqZcyVej3vHSE3JdoA53In_CGZplr3iujWMpq3UGWoutRwTWADxtdBEaXC0vHh9kyGgrS8uiTMA',
      OPENAI_MODEL: 'gpt-4o-mini',
      OPENAI_MAX_TOKENS: '400',
      OPENAI_TEMPERATURE: '0.0',
      DATABASE_URL: 'postgresql://root:Ai_ats@2000@147.93.155.233:5432/ai_ats',
      DEBUG: 'False'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
}
