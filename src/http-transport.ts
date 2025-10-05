import express, { Request, Response } from 'express';
import cors from 'cors';
import https from 'https';
import http from 'http';
import fs from 'fs';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';

export interface HttpTransportConfig {
  port: number;
  host?: string;
  cors?: boolean;
  https?: {
    certPath: string;
    keyPath: string;
  };
}

/**
 * HTTP/SSE transport for remote MCP access
 * Simplified version for local network use (no authentication)
 */
export class HttpTransport {
  private app: express.Application;
  private server: any;
  private mcpServer: Server;
  private clients: Set<Response> = new Set();

  constructor(
    private config: HttpTransportConfig,
    mcpServer: Server
  ) {
    this.mcpServer = mcpServer;
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware(): void {
    // Enable CORS for local network access
    if (this.config.cors !== false) {
      this.app.use(cors({
        origin: '*', // Allow all origins for local network
        methods: ['GET', 'POST', 'OPTIONS'],
        allowedHeaders: ['Content-Type'],
      }));
    }

    // Parse JSON bodies
    this.app.use(express.json());

    // Request logging
    this.app.use((req, res, next) => {
      console.log(`[HTTP] ${req.method} ${req.path}`);
      next();
    });
  }

  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'ok',
        transport: 'http',
        timestamp: new Date().toISOString(),
      });
    });

    // SSE endpoint for streaming responses
    this.app.get('/sse', (req, res) => {
      console.log('[HTTP] Client connected via SSE');

      // Set SSE headers
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Add client to active connections
      this.clients.add(res);

      // Send initial connection message
      res.write('data: {"type":"connected"}\n\n');

      // Remove client on disconnect
      req.on('close', () => {
        console.log('[HTTP] Client disconnected from SSE');
        this.clients.delete(res);
      });
    });

    // JSON-RPC endpoint for client requests
    this.app.post('/message', async (req, res) => {
      try {
        const message = req.body;
        console.log('[HTTP] Received message:', JSON.stringify(message).substring(0, 100));

        // In a full implementation, this would route to the MCP server
        // For now, return a basic response structure
        res.json({
          jsonrpc: '2.0',
          id: message.id,
          result: {
            protocolVersion: '2025-03-26',
            capabilities: {},
            serverInfo: {
              name: 'local-llm-mcp-server',
              version: '1.0.0',
            },
          },
        });
      } catch (error) {
        console.error('[HTTP] Error handling message:', error);
        res.status(500).json({
          jsonrpc: '2.0',
          id: req.body?.id || null,
          error: {
            code: -32603,
            message: 'Internal error',
            data: error instanceof Error ? error.message : 'Unknown error',
          },
        });
      }
    });

    // Info endpoint
    this.app.get('/', (req, res) => {
      res.json({
        server: 'local-llm-mcp-server',
        version: '1.0.0',
        transport: 'http',
        endpoints: {
          health: '/health',
          sse: '/sse',
          message: '/message',
        },
        documentation: 'https://github.com/yourusername/local-llm-mcp-server',
      });
    });
  }

  /**
   * Broadcast a message to all connected SSE clients
   */
  private broadcast(message: any): void {
    const data = JSON.stringify(message);
    this.clients.forEach(client => {
      client.write(`data: ${data}\n\n`);
    });
  }

  /**
   * Start the HTTP or HTTPS server
   */
  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const host = this.config.host || '0.0.0.0';
        const protocol = this.config.https ? 'https' : 'http';

        if (this.config.https) {
          // HTTPS mode
          const { certPath, keyPath } = this.config.https;

          if (!certPath || !keyPath) {
            throw new Error('HTTPS enabled but certificate or key path not provided');
          }

          if (!fs.existsSync(certPath)) {
            throw new Error(`Certificate file not found: ${certPath}`);
          }

          if (!fs.existsSync(keyPath)) {
            throw new Error(`Key file not found: ${keyPath}`);
          }

          const httpsOptions = {
            cert: fs.readFileSync(certPath),
            key: fs.readFileSync(keyPath)
          };

          this.server = https.createServer(httpsOptions, this.app);
        } else {
          // HTTP mode
          this.server = http.createServer(this.app);
        }

        this.server.listen(this.config.port, host, () => {
          console.log(`[${protocol.toUpperCase()}] MCP Server listening on ${protocol}://${host}:${this.config.port}`);
          console.log(`[${protocol.toUpperCase()}] SSE endpoint: ${protocol}://${host}:${this.config.port}/sse`);
          console.log(`[${protocol.toUpperCase()}] Message endpoint: ${protocol}://${host}:${this.config.port}/message`);
          resolve();
        });

        this.server.on('error', (error: Error) => {
          console.error(`[${protocol.toUpperCase()}] Server error:`, error);
          reject(error);
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Stop the HTTP server
   */
  async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        // Close all SSE connections
        this.clients.forEach(client => {
          client.end();
        });
        this.clients.clear();

        this.server.close(() => {
          console.log('[HTTP] Server stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  /**
   * Get the number of active connections
   */
  getConnectionCount(): number {
    return this.clients.size;
  }
}
