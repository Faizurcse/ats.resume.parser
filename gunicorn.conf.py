# Gunicorn configuration file for AI ATS Python Backend

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 6  # 3 CPU cores × 2 = 6 workers
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "./logs/access.log"
errorlog = "./logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'ai_ats_python_backend'

# Server mechanics
daemon = False
pidfile = './gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Performance
preload_app = True
reload = False
