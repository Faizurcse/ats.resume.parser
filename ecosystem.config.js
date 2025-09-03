module.exports = {
  apps: [{
    name: 'AI_Ats_python_Backend',
    script: 'python',
    args: 'run.py',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PORT: '8000',
      OPENAI_API_KEY: process.env.OPENAI_API_KEY || '',
      OPENAI_MODEL: 'gpt-4o-mini',
      OPENAI_MAX_TOKENS: '400',
      OPENAI_TEMPERATURE: '0.0',
      DATABASE_URL: 'postgresql://root:Ai_ats%402000@147.93.155.233:5432/ai_ats',
      DEBUG: 'False'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
}