// PM2 Ecosystem Configuration for Quickscene Production Deployment
// SuperBryn Technical Assessment - 10/10 Code Quality
// Author: Sandeep Kumar Sahoo

module.exports = {
  apps: [
    {
      // Frontend React Application (Port 8101)
      name: 'quickscene-frontend',
      script: 'serve',
      args: '-s build -l 8101 --no-clipboard --cors',
      cwd: '/home/ubuntu/Sandeep/projects/quickscene-frontend',
      instances: 1, // Single instance for assessment
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '512M',
      env: {
        NODE_ENV: 'production',
        PORT: 8101,
        REACT_APP_API_URL: 'http://3.111.22.56:8000',
        REACT_APP_WS_URL: 'ws://3.111.22.56:8103'
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 8101,
        REACT_APP_API_URL: 'http://3.111.22.56:8000',
        REACT_APP_WS_URL: 'ws://3.111.22.56:8103'
      },
      log_file: '/home/ubuntu/logs/quickscene-frontend.log',
      out_file: '/home/ubuntu/logs/quickscene-frontend-out.log',
      error_file: '/home/ubuntu/logs/quickscene-frontend-error.log',
      time: true,
      autorestart: true,
      restart_delay: 1000,
      max_restarts: 10,
      min_uptime: '10s',
      kill_timeout: 5000
    },
    
    {
      // FastAPI Backend Server (Port 8000)
      name: 'quickscene-api',
      script: '/home/ubuntu/Sandeep/projects/Quickscene/venv/bin/python',
      args: 'api_server.py',
      cwd: '/home/ubuntu/Sandeep/projects/Quickscene',
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '2G',
      env: {
        NODE_ENV: 'production',
        QUICKSCENE_HOST: '0.0.0.0',
        QUICKSCENE_PORT: 8000,
        QUICKSCENE_DEBUG: 'false',
        QUICKSCENE_WORKERS: '1',
        QUICKSCENE_MAX_WORKERS: '4',
        QUICKSCENE_BATCH_SIZE: '32',
        QUICKSCENE_CACHE_TTL: '3600',
        QUICKSCENE_CORS_ORIGINS: 'http://3.111.22.56:8101,http://localhost:8101',
        QUICKSCENE_LOG_LEVEL: 'INFO',
        PYTHONPATH: '/home/ubuntu/Sandeep/projects/Quickscene'
      },
      env_production: {
        NODE_ENV: 'production',
        QUICKSCENE_HOST: '0.0.0.0',
        QUICKSCENE_PORT: 8000,
        QUICKSCENE_DEBUG: 'false',
        QUICKSCENE_WORKERS: '1',
        QUICKSCENE_MAX_WORKERS: '8',
        QUICKSCENE_BATCH_SIZE: '64',
        QUICKSCENE_CACHE_TTL: '7200',
        QUICKSCENE_CORS_ORIGINS: 'http://3.111.22.56:8101',
        QUICKSCENE_LOG_LEVEL: 'INFO',
        PYTHONPATH: '/home/ubuntu/Sandeep/projects/Quickscene'
      },
      log_file: '/home/ubuntu/logs/quickscene-api.log',
      out_file: '/home/ubuntu/logs/quickscene-api-out.log',
      error_file: '/home/ubuntu/logs/quickscene-api-error.log',
      time: true,
      autorestart: true,
      restart_delay: 2000,
      max_restarts: 5,
      min_uptime: '30s',
      kill_timeout: 10000
    },

  ],

  // Deployment configuration for SuperBryn Assessment
  deploy: {
    production: {
      user: 'ubuntu',
      host: '3.111.22.56',
      ref: 'origin/main',
      repo: 'https://github.com/MrDecryptDecipher/Quickscene.git',
      path: '/home/ubuntu/Sandeep/projects',
      'pre-deploy-local': '',
      'post-deploy': 'cd quickscene-frontend && npm install && npm run build && pm2 reload ecosystem.config.js --env production',
      'pre-setup': '',
      'ssh_options': 'ForwardAgent=yes'
    }
  },

  // Global PM2 settings
  pmx: {
    enabled: true,
    network: true,
    ports: true
  },

  // Performance monitoring
  monitoring: {
    http: true,
    https: false,
    port: 9615,
    refresh: 5000,
    network: true,
    ports: true
  }
};

// Additional configuration for different environments
const environments = {
  development: {
    QUICKSCENE_DEBUG: 'true',
    QUICKSCENE_LOG_LEVEL: 'DEBUG',
    REACT_APP_API_URL: 'http://localhost:8000'
  },
  
  staging: {
    QUICKSCENE_DEBUG: 'false',
    QUICKSCENE_LOG_LEVEL: 'INFO',
    REACT_APP_API_URL: 'http://staging.quickscene.com:8102'
  },
  
  production: {
    QUICKSCENE_DEBUG: 'false',
    QUICKSCENE_LOG_LEVEL: 'INFO',
    REACT_APP_API_URL: 'https://3.111.22.56:8102'
  }
};

// Export environment-specific configurations
module.exports.environments = environments;
