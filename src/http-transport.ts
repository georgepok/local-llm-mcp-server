import express, { Request, Response } from 'express';
import cors from 'cors';
import https from 'https';
import http from 'http';
import fs from 'fs';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';

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
 * Uses MCP SDK's SSEServerTransport for proper protocol handling
 */
export class HttpTransport {
  private app: express.Application;
  private server: any;
  private mcpServer: Server;
  private transports: Map<string, SSEServerTransport> = new Map();

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

    // SSE endpoint - establishes SSE connection for MCP protocol
    this.app.get('/sse', async (req, res) => {
      console.log('[HTTP] Client connecting via SSE');

      // Create SSE transport for this client
      const transport = new SSEServerTransport('/message', res);

      // Store transport by session ID
      this.transports.set(transport.sessionId, transport);

      // Connect the MCP server to this transport
      await this.mcpServer.connect(transport);

      console.log(`[HTTP] Client connected via SSE (session: ${transport.sessionId})`);

      // Clean up on disconnect
      transport.onclose = () => {
        console.log(`[HTTP] Client disconnected from SSE (session: ${transport.sessionId})`);
        this.transports.delete(transport.sessionId);
      };
    });

    // Message endpoint - receives JSON-RPC messages from clients
    this.app.post('/message', async (req, res) => {
      try {
        console.log('[HTTP] Received message:', JSON.stringify(req.body).substring(0, 100));

        // Extract session ID from query parameter
        const sessionId = req.query.sessionId as string;

        if (!sessionId) {
          res.status(400).json({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32600,
              message: 'Invalid Request: sessionId required',
            },
          });
          return;
        }

        // Find the transport for this session
        const transport = this.transports.get(sessionId);

        if (!transport) {
          res.status(404).json({
            jsonrpc: '2.0',
            id: null,
            error: {
              code: -32001,
              message: 'Session not found',
            },
          });
          return;
        }

        // Forward the message to the transport
        await transport.handlePostMessage(req, res, req.body);

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
        // Close all transports
        this.transports.forEach(transport => {
          transport.close();
        });
        this.transports.clear();

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
    return this.transports.size;
  }
}
