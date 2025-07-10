# 🚀 Quickscene Production Deployment Guide

> **SuperBryn Technical Assessment - Production Deployment on 3.111.22.56**

## 📋 **Overview**

This guide covers the complete production deployment of the Quickscene video search system for the SuperBryn technical assessment, including frontend (React), backend (FastAPI), and infrastructure configuration.

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    3.111.22.56 (Production Server)          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Nginx     │  │   React     │  │   FastAPI   │         │
│  │   (80/443)  │  │   (8101)    │  │   (8102)    │         │
│  │   Proxy     │  │   Frontend  │  │   Backend   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   PM2       │  │   SSL/TLS   │  │  WebSocket  │         │
│  │   Manager   │  │   Certs     │  │   (8103)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 **Deployment Targets**

- **Frontend**: React TypeScript app on port 8101
- **Backend**: FastAPI server on port 8102  
- **WebSocket**: Real-time features on port 8103
- **Proxy**: Nginx reverse proxy on ports 80/443
- **SSL**: HTTPS with Let's Encrypt certificates
- **Process Management**: PM2 for auto-restart and monitoring

## 📦 **Prerequisites**

### **System Requirements**
- Ubuntu 20.04+ LTS
- 4GB+ RAM
- 20GB+ storage
- Node.js 18+
- Python 3.12+
- Nginx
- PM2

### **Pre-deployment Checklist**
- [ ] Server access via SSH
- [ ] Domain/IP configured (3.111.22.56)
- [ ] Firewall ports opened (80, 443, 8101, 8102, 8103)
- [ ] Git repository access
- [ ] SSL certificate ready

## 🚀 **Quick Deployment**

### **1. One-Command Deployment**
```bash
# Clone and deploy in one step
git clone https://github.com/MrDecryptDecipher/Quickscene.git
cd quickscene-frontend
chmod +x deploy.sh
./deploy.sh deploy
```

### **2. Manual Step-by-Step Deployment**

#### **Step 1: System Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install PM2 globally
npm install -g pm2

# Install Nginx
sudo apt install -y nginx

# Install Python dependencies (if not already done)
sudo apt install -y python3-pip python3-venv
```

#### **Step 2: Project Setup**
```bash
# Navigate to project directory
cd /home/ubuntu/Sandeep/projects/quickscene-frontend

# Install frontend dependencies
npm ci

# Build production frontend
REACT_APP_API_URL="http://3.111.22.56:8102" npm run build

# Setup backend (if not already done)
cd ../Quickscene
source venv/bin/activate
pip install -r requirements.txt
```

#### **Step 3: Configure Services**
```bash
# Copy Nginx configuration
sudo cp /home/ubuntu/Sandeep/projects/quickscene-frontend/nginx.conf /etc/nginx/nginx.conf

# Test Nginx configuration
sudo nginx -t

# Start PM2 services
cd /home/ubuntu/Sandeep/projects/quickscene-frontend
pm2 start ecosystem.config.js --env production
pm2 save
pm2 startup
```

#### **Step 4: Start Services**
```bash
# Restart Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx

# Check service status
pm2 status
sudo systemctl status nginx
```

## 🔧 **Configuration Files**

### **Nginx Configuration** (`nginx.conf`)
- Reverse proxy for all services
- SSL termination
- Rate limiting
- CORS headers
- Static file optimization
- Security headers

### **PM2 Ecosystem** (`ecosystem.config.js`)
- Multi-service management
- Environment variables
- Auto-restart policies
- Log management
- Cluster mode for frontend

### **Environment Variables**
```bash
# Production Environment
NODE_ENV=production
REACT_APP_API_URL=https://3.111.22.56:8102
QUICKSCENE_HOST=0.0.0.0
QUICKSCENE_PORT=8102
QUICKSCENE_DEBUG=false
QUICKSCENE_CORS_ORIGINS=https://3.111.22.56
```

## 📊 **Service Management**

### **PM2 Commands**
```bash
# View all processes
pm2 status

# View logs
pm2 logs

# Restart all services
pm2 restart all

# Stop all services
pm2 stop all

# Monitor in real-time
pm2 monit

# Reload with zero downtime
pm2 reload all
```

### **Nginx Commands**
```bash
# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Check status
sudo systemctl status nginx

# View logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## 🔍 **Health Checks & Monitoring**

### **Service Health Endpoints**
```bash
# Frontend health
curl https://3.111.22.56/

# API health
curl https://3.111.22.56/api/v1/health

# System status
curl https://3.111.22.56/api/v1/status

# Metrics (internal only)
curl http://localhost:8102/metrics
```

### **Performance Monitoring**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s https://3.111.22.56/api/v1/health

# Monitor resource usage
htop
pm2 monit

# Check port usage
netstat -tuln | grep -E ':(80|443|8101|8102|8103)'
```

## 🛡️ **Security Configuration**

### **SSL/TLS Setup**
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Generate Let's Encrypt certificate
sudo certbot --nginx -d 3.111.22.56

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **Firewall Configuration**
```bash
# Configure UFW
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8101
sudo ufw allow 8102
sudo ufw allow 8103
sudo ufw enable
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Frontend Not Loading**
```bash
# Check PM2 status
pm2 status

# Check frontend logs
pm2 logs quickscene-frontend

# Restart frontend
pm2 restart quickscene-frontend
```

#### **API Not Responding**
```bash
# Check API logs
pm2 logs quickscene-api

# Check Python environment
cd /home/ubuntu/Sandeep/projects/Quickscene
source venv/bin/activate
python -c "import fastapi; print('FastAPI OK')"

# Restart API
pm2 restart quickscene-api
```

#### **Nginx Issues**
```bash
# Check Nginx configuration
sudo nginx -t

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log

# Restart Nginx
sudo systemctl restart nginx
```

#### **SSL Certificate Issues**
```bash
# Check certificate status
sudo certbot certificates

# Renew certificates
sudo certbot renew

# Test SSL configuration
openssl s_client -connect 3.111.22.56:443
```

### **Performance Issues**
```bash
# Check system resources
free -h
df -h
top

# Check PM2 memory usage
pm2 monit

# Restart services if needed
pm2 restart all
```

## 📈 **Performance Optimization**

### **Frontend Optimization**
- Code splitting enabled
- Gzip compression
- Static asset caching
- CDN-ready configuration

### **Backend Optimization**
- Connection pooling
- Response caching
- Database query optimization
- Memory management

### **Infrastructure Optimization**
- Nginx caching
- Load balancing ready
- SSL session caching
- Keep-alive connections

## 📋 **Deployment Checklist**

### **Pre-Deployment**
- [ ] Server provisioned and accessible
- [ ] Domain/IP configured
- [ ] SSL certificates ready
- [ ] Firewall configured
- [ ] Dependencies installed

### **Deployment**
- [ ] Code deployed and built
- [ ] Services configured
- [ ] PM2 processes started
- [ ] Nginx configured and running
- [ ] SSL certificates installed

### **Post-Deployment**
- [ ] Health checks passing
- [ ] Performance tests completed
- [ ] Monitoring configured
- [ ] Backup procedures in place
- [ ] Documentation updated

## 🎯 **Assessment URLs**

After successful deployment, the following URLs will be available:

- **Main Application**: https://3.111.22.56
- **API Documentation**: https://3.111.22.56/api/docs
- **Health Check**: https://3.111.22.56/api/v1/health
- **System Status**: https://3.111.22.56/api/v1/status
- **GitHub Repository**: https://github.com/MrDecryptDecipher/Quickscene

## 📞 **Support**

For deployment issues or questions:
- Check logs: `pm2 logs` and `/var/log/nginx/`
- Monitor services: `pm2 monit`
- Review configuration: `nginx.conf` and `ecosystem.config.js`

---

**Deployment Guide Version**: 1.0.0  
**Last Updated**: July 9, 2025  
**Author**: Sandeep Kumar Sahoo  
**Assessment**: SuperBryn Technical Assessment
