# Script Command-Line Options

All server startup scripts now support command-line arguments for easy configuration.

## Quick Examples

```bash
# Start HTTP server on port 8080
./scripts/start-remote.sh --port 8080

# Start HTTPS server on port 8443
./scripts/start-https.sh --port 8443

# Start dual mode on port 5000 with HTTPS
./scripts/start-dual.sh --port 5000 --https

# Custom host and port
./scripts/start-remote.sh --host 192.168.1.100 --port 9000
```

---

## start-remote.sh (HTTP Mode)

**Synopsis:**
```bash
./scripts/start-remote.sh [OPTIONS]
```

**Options:**

| Option | Short | Description | Default | Example |
|--------|-------|-------------|---------|---------|
| `--port` | `-p` | Port number | 3000 | `--port 8080` |
| `--host` | `-h` | Host address | 0.0.0.0 | `--host 192.168.1.100` |

**Examples:**
```bash
# Port 8080
./scripts/start-remote.sh --port 8080

# Custom host and port
./scripts/start-remote.sh -h 192.168.1.100 -p 9000

# Environment variables still work
PORT=8080 ./scripts/start-remote.sh

# Via npm (pass after --)
npm run start:remote -- --port 8080
```

---

## start-https.sh (HTTPS Mode)

**Synopsis:**
```bash
./scripts/start-https.sh [OPTIONS]
```

**Options:**

| Option | Short | Description | Default | Example |
|--------|-------|-------------|---------|---------|
| `--port` | `-p` | Port number | 3000 | `--port 8443` |
| `--host` | `-h` | Host address | 0.0.0.0 | `--host 192.168.1.100` |
| `--cert` | - | Certificate path | ./certs/cert.pem | `--cert /path/to/cert.pem` |
| `--key` | - | Key path | ./certs/key.pem | `--key /path/to/key.pem` |

**Examples:**
```bash
# Port 8443
./scripts/start-https.sh --port 8443

# Custom certificates
./scripts/start-https.sh --cert /etc/ssl/cert.pem --key /etc/ssl/key.pem

# All options
./scripts/start-https.sh -p 443 -h 0.0.0.0 --cert /etc/letsencrypt/live/domain/fullchain.pem --key /etc/letsencrypt/live/domain/privkey.pem

# Via npm
npm run start:https -- --port 8443
```

---

## start-dual.sh (Dual Mode)

**Synopsis:**
```bash
./scripts/start-dual.sh [OPTIONS]
```

**Options:**

| Option | Short | Description | Default | Example |
|--------|-------|-------------|---------|---------|
| `--port` | `-p` | Port number | 3000 | `--port 5000` |
| `--host` | `-h` | Host address | 0.0.0.0 | `--host 192.168.1.100` |
| `--https` | - | Enable HTTPS mode | false | `--https` |
| `--cert` | - | Certificate path | ./certs/cert.pem | `--cert /path/to/cert.pem` |
| `--key` | - | Key path | ./certs/key.pem | `--key /path/to/key.pem` |

**Examples:**
```bash
# Dual mode with HTTP on port 8080
./scripts/start-dual.sh --port 8080

# Dual mode with HTTPS
./scripts/start-dual.sh --https

# Dual mode with HTTPS on custom port
./scripts/start-dual.sh --port 8443 --https

# Custom certificates
./scripts/start-dual.sh --https --cert /etc/ssl/cert.pem --key /etc/ssl/key.pem

# All options
./scripts/start-dual.sh -p 5000 -h 0.0.0.0 --https --cert /path/to/cert.pem --key /path/to/key.pem

# Via npm
npm run start:dual -- --port 5000 --https
```

---

## Combining Options

You can combine multiple options:

```bash
# HTTP mode
./scripts/start-remote.sh -p 8080 -h 192.168.1.100

# HTTPS mode
./scripts/start-https.sh -p 8443 --cert /etc/ssl/cert.pem --key /etc/ssl/key.pem

# Dual mode
./scripts/start-dual.sh -p 5000 --https --cert /custom/cert.pem --key /custom/key.pem
```

---

## Using with npm Scripts

To pass arguments through npm, use `--` separator:

```bash
# HTTP
npm run start:remote -- --port 8080

# HTTPS
npm run start:https -- --port 8443 --cert /path/to/cert.pem

# Dual
npm run start:dual -- --port 5000 --https
```

---

## Priority Order

Arguments are applied in this order (later overrides earlier):

1. Script defaults (hardcoded)
2. Environment variables
3. Command-line arguments (highest priority)

**Example:**
```bash
# Environment variable sets port to 7000
export PORT=7000

# Command-line argument overrides to 8080
./scripts/start-remote.sh --port 8080

# Result: Server starts on port 8080
```

---

## Common Use Cases

### Development - Different Ports for Testing

```bash
# Terminal 1: HTTP on 3000
./scripts/start-remote.sh --port 3000

# Terminal 2: HTTPS on 3001 (for testing)
./scripts/start-https.sh --port 3001
```

### Production - Standard HTTPS Port

```bash
# Run on port 443 (requires sudo)
sudo ./scripts/start-https.sh --port 443 \
  --cert /etc/letsencrypt/live/domain.com/fullchain.pem \
  --key /etc/letsencrypt/live/domain.com/privkey.pem
```

### Local Network - Custom IP Binding

```bash
# Bind to specific network interface
./scripts/start-remote.sh --host 192.168.1.100 --port 8080
```

### Dual Mode - Both Local and Remote

```bash
# Claude Desktop (stdio) + Network (HTTPS on 8443)
./scripts/start-dual.sh --port 8443 --https
```

---

## Error Handling

**Invalid option:**
```bash
./scripts/start-remote.sh --invalid
# Output: Unknown option: --invalid
#         Usage: ./scripts/start-remote.sh [--port PORT] [--host HOST]
```

**Missing argument value:**
```bash
./scripts/start-remote.sh --port
# Output: (port value will be empty, may cause error)
```

**Port already in use:**
```bash
./scripts/start-remote.sh --port 3000
# If port 3000 is busy:
# Error: EADDRINUSE: address already in use :::3000

# Solution: Use different port
./scripts/start-remote.sh --port 3001
```

---

## Help

Each script shows usage when invalid options are provided:

```bash
./scripts/start-remote.sh --help
# Shows: Usage: ./scripts/start-remote.sh [--port PORT] [--host HOST]

./scripts/start-https.sh --help
# Shows: Usage: ./scripts/start-https.sh [--port PORT] [--host HOST] [--cert CERT_PATH] [--key KEY_PATH]

./scripts/start-dual.sh --help
# Shows: Usage: ./scripts/start-dual.sh [--port PORT] [--host HOST] [--https] [--cert CERT_PATH] [--key KEY_PATH]
```

---

## Environment Variables (Still Supported)

All environment variables continue to work:

```bash
# Environment variable method
PORT=8080 ./scripts/start-remote.sh

# Command-line argument method (same result)
./scripts/start-remote.sh --port 8080

# Combined (command-line takes priority)
PORT=7000 ./scripts/start-remote.sh --port 8080
# Result: Port 8080 is used
```

**Available environment variables:**
- `PORT` - Server port
- `HOST` - Bind address
- `HTTPS_CERT_PATH` - SSL certificate path
- `HTTPS_KEY_PATH` - SSL key path
- `USE_HTTPS` - Enable HTTPS in dual mode (true/false)

---

## Summary

**Quick Reference:**

```bash
# HTTP on custom port
./scripts/start-remote.sh -p 8080

# HTTPS on custom port
./scripts/start-https.sh -p 8443

# Dual mode with HTTPS
./scripts/start-dual.sh --https -p 5000

# Custom everything
./scripts/start-https.sh -p 443 -h 0.0.0.0 --cert /path/cert.pem --key /path/key.pem
```

All scripts support both command-line arguments and environment variables for maximum flexibility!
