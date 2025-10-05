# Deployment Guide

This guide covers deployment scenarios for the Local LLM MCP Server across different environments and use cases.

## Prerequisites

- Node.js 18+ installed
- LM Studio with a compatible model
- MCP-compatible host application (Claude Desktop, etc.)

## Local Development Setup

### 1. LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (e.g., Llama 3.1, Code Llama, etc.)
3. Load the model and start the local server:
   - Click "Local Server" tab
   - Select your model
   - Start server on port 1234 (default)

### 2. MCP Server Setup

```bash
# Clone and setup
git clone <repository-url>
cd local-llm-mcp-server

# Install dependencies
npm install

# Build the project
npm run build

# Create default configuration
cp examples/config.json config.json

# Start the server
npm start
```

### 3. Claude Desktop Integration

Copy the MCP server configuration:

```bash
# macOS
cp examples/claude-desktop-config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
cp examples/claude-desktop-config.json ~/.config/claude/claude_desktop_config.json

# Windows
copy examples\claude-desktop-config.json %APPDATA%\Claude\claude_desktop_config.json
```

Update the path in the configuration file to match your installation directory.

## Production Deployment

### Server Environment

#### System Requirements

- **Memory**: 8GB RAM minimum (16GB+ recommended for larger models)
- **Storage**: 20GB+ free space for models and cache
- **CPU**: Modern multi-core processor (Apple Silicon or x86_64)
- **Network**: Stable internet for initial model downloads

#### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY dist/ ./dist/
COPY config.json ./

# Create user for security
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001
USER nodejs

EXPOSE 3000

CMD ["npm", "start"]
```

Build and run:

```bash
# Build image
docker build -t local-llm-mcp .

# Run container
docker run -d \
  --name local-llm-mcp \
  -p 3000:3000 \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/models:/app/models \
  local-llm-mcp
```

#### Process Management with PM2

```bash
# Install PM2 globally
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'local-llm-mcp',
    script: 'dist/index.js',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    env: {
      NODE_ENV: 'production',
      LLM_SERVER_PORT: '1234'
    }
  }]
};
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### High Availability Setup

#### Load Balancer Configuration (nginx)

```nginx
upstream local_llm_backend {
    server localhost:3001;
    server localhost:3002;
    server localhost:3003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://local_llm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_timeout 300s;
    }
}
```

#### Multiple LM Studio Instances

Run multiple LM Studio instances on different ports:

```bash
# Instance 1 - Port 1234
# Instance 2 - Port 1235
# Instance 3 - Port 1236
```

Configure load balancing in `config.json`:

```json
{
  "lmStudio": {
    "instances": [
      {
        "baseUrl": "http://localhost:1234/v1",
        "weight": 1
      },
      {
        "baseUrl": "http://localhost:1235/v1",
        "weight": 1
      },
      {
        "baseUrl": "http://localhost:1236/v1",
        "weight": 1
      }
    ]
  }
}
```

## Security Considerations

### Network Security

```bash
# Use firewall to restrict access
sudo ufw allow from 192.168.1.0/24 to any port 1234
sudo ufw deny 1234

# Use reverse proxy with SSL
certbot --nginx -d your-domain.com
```

### Configuration Security

```json
{
  "security": {
    "enableCORS": false,
    "allowedOrigins": ["https://claude.ai"],
    "rateLimiting": {
      "windowMs": 900000,
      "max": 100
    },
    "authentication": {
      "enabled": true,
      "method": "api_key"
    }
  }
}
```

### Data Protection

```json
{
  "privacy": {
    "defaultLevel": "strict",
    "enableLogging": false,
    "encryptCache": true,
    "automaticCleanup": true,
    "retentionPolicy": {
      "logs": 0,
      "cache": 3600,
      "sessions": 86400
    }
  }
}
```

## Monitoring and Observability

### Health Checks

```bash
# Basic health check endpoint
curl -f http://localhost:3000/health || exit 1

# Detailed status check
curl http://localhost:3000/status
```

### Logging Configuration

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "destinations": [
      {
        "type": "file",
        "path": "/var/log/local-llm-mcp/app.log",
        "maxFiles": 5,
        "maxSize": "10m"
      },
      {
        "type": "console",
        "colorize": false
      }
    ]
  }
}
```

### Metrics Collection

```javascript
// Add to index.ts for basic metrics
const metrics = {
  requestCount: 0,
  errorCount: 0,
  averageResponseTime: 0,
  activeConnections: 0
};

// Export metrics endpoint
app.get('/metrics', (req, res) => {
  res.json(metrics);
});
```

## Performance Optimization

### Model Selection

Choose models based on your hardware:

```json
{
  "modelRecommendations": {
    "8GB_RAM": ["llama-3.1-8b-instruct", "codellama-7b"],
    "16GB_RAM": ["llama-3.1-70b-instruct", "codellama-13b"],
    "32GB_RAM": ["llama-3.1-405b-instruct", "codellama-34b"]
  }
}
```

### Caching Strategy

```json
{
  "cache": {
    "enabled": true,
    "type": "redis",
    "connection": {
      "host": "localhost",
      "port": 6379,
      "db": 0
    },
    "ttl": {
      "responses": 3600,
      "templates": 86400,
      "models": 604800
    }
  }
}
```

### Connection Pooling

```json
{
  "performance": {
    "connectionPool": {
      "maxConnections": 10,
      "minConnections": 2,
      "acquireTimeoutMillis": 30000,
      "idleTimeoutMillis": 30000
    },
    "requestQueue": {
      "maxSize": 100,
      "priorityLevels": 3
    }
  }
}
```

## Backup and Recovery

### Configuration Backup

```bash
#!/bin/bash
# backup-config.sh

BACKUP_DIR="/backup/local-llm-mcp"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration
cp config.json $BACKUP_DIR/config_$DATE.json

# Backup custom prompts
cp -r prompts/ $BACKUP_DIR/prompts_$DATE/

# Backup logs
cp -r logs/ $BACKUP_DIR/logs_$DATE/

echo "Backup completed: $BACKUP_DIR"
```

### Model Backup

```bash
#!/bin/bash
# backup-models.sh

LM_STUDIO_DIR="$HOME/.cache/lm-studio"
BACKUP_DIR="/backup/lm-studio-models"

rsync -av --progress $LM_STUDIO_DIR/models/ $BACKUP_DIR/
```

### Automated Backup Schedule

```cron
# Add to crontab
0 2 * * * /opt/local-llm-mcp/scripts/backup-config.sh
0 3 * * 0 /opt/local-llm-mcp/scripts/backup-models.sh
```

## Troubleshooting

### Common Issues

**Connection Refused**
```bash
# Check LM Studio is running
curl http://localhost:1234/v1/models

# Check MCP server status
ps aux | grep "local-llm-mcp"

# Check logs
tail -f logs/app.log
```

**High Memory Usage**
```bash
# Monitor memory usage
htop

# Check model size
du -sh ~/.cache/lm-studio/models/*

# Adjust model parameters
# Reduce max_tokens, batch_size in config
```

**Slow Response Times**
```bash
# Check system resources
iostat -x 1

# Monitor GPU usage (if applicable)
nvidia-smi

# Check network latency
ping localhost
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=mcp:* npm start

# Verbose LM Studio logs
LM_STUDIO_LOG_LEVEL=debug npm start

# Profile performance
NODE_ENV=profiling npm start
```

### Recovery Procedures

1. **Service Recovery**
   ```bash
   # Restart MCP server
   pm2 restart local-llm-mcp

   # Restart LM Studio
   killall "LM Studio"
   open -a "LM Studio"
   ```

2. **Configuration Recovery**
   ```bash
   # Restore from backup
   cp /backup/local-llm-mcp/config_latest.json config.json

   # Validate configuration
   npm run validate-config
   ```

3. **Model Recovery**
   ```bash
   # Re-download model
   # Use LM Studio interface to re-download

   # Or restore from backup
   rsync -av /backup/lm-studio-models/ ~/.cache/lm-studio/models/
   ```

This deployment guide ensures reliable, secure, and performant operation of the Local LLM MCP Server in various environments.