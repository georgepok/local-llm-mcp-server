import express from 'express';
import cors from 'cors';
import https from 'https';
import http from 'http';
import fs from 'fs';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { randomUUID } from 'crypto';

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
 * Streamable HTTP transport for MCP
 * Implements MCP Streamable HTTP Transport Specification (2025-03-26)
 *
 * Features:
 * - Single /mcp endpoint for GET, POST, and DELETE
 * - Stateful session management with session IDs
 * - SSE streaming for server messages
 * - CORS support for network access
 */
export class HttpTransport {
  private app: express.Application;
  private server: http.Server | https.Server | null = null;
  private mcpServer: Server;
  private transport: StreamableHTTPServerTransport | null = null;

  constructor(
    private config: HttpTransportConfig,
    mcpServer: Server
  ) {
    this.mcpServer = mcpServer;
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
  }

  /**
   * Configure Express middleware
   */
  private setupMiddleware(): void {
    // Enable CORS if not explicitly disabled
    if (this.config.cors !== false) {
      this.app.use(cors({
        origin: '*',
        methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
        allowedHeaders: [
          'Content-Type',
          'Accept',
          'Mcp-Protocol-Version',
          'Mcp-Session-Id',
          'Last-Event-ID',
        ],
        exposedHeaders: ['Mcp-Session-Id', 'Mcp-Protocol-Version'],
      }));
    }

    // Parse JSON request bodies
    this.app.use(express.json());

    // Request logging (to stderr per MCP spec)
    this.app.use((req, res, next) => {
      console.error(`[HTTP] ${req.method} ${req.path}`);
      next();
    });
  }

  /**
   * Setup HTTP routes
   */
  private setupRoutes(): void {
    // Health check endpoint (non-MCP)
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'ok',
        transport: 'streamable-http',
        protocol: '2025-03-26',
        timestamp: new Date().toISOString(),
      });
    });

    // MCP endpoint - handles all MCP protocol messages
    // Supports: GET (SSE stream), POST (JSON-RPC), DELETE (session termination)
    this.app.all('/mcp', async (req, res) => {
      if (!this.transport) {
        res.status(503).json({
          jsonrpc: '2.0',
          error: {
            code: -32000,
            message: 'Transport not initialized',
          },
          id: null,
        });
        return;
      }

      try {
        // Delegate to StreamableHTTPServerTransport
        // It handles all protocol requirements: headers, session management, SSE, etc.
        await this.transport.handleRequest(req, res, req.body);
      } catch (error) {
        console.error('[HTTP] Error handling request:', error);

        // Only send error if headers haven't been sent (SSE streams send headers immediately)
        if (!res.headersSent) {
          res.status(500).json({
            jsonrpc: '2.0',
            error: {
              code: -32603,
              message: 'Internal error',
              data: error instanceof Error ? error.message : 'Unknown error',
            },
            id: null,
          });
        }
      }
    });

    // Server info endpoint (non-MCP)
    this.app.get('/', (req, res) => {
      res.json({
        server: 'local-llm-mcp-server',
        version: '1.0.0',
        transport: 'streamable-http',
        protocol: '2025-03-26',
        endpoints: {
          health: '/health',
          mcp: '/mcp',
        },
        documentation: 'https://github.com/yourusername/local-llm-mcp-server',
      });
    });
  }

  /**
   * Start the HTTP or HTTPS server
   */
  async start(): Promise<void> {
    const host = this.config.host || '0.0.0.0';
    const protocol = this.config.https ? 'https' : 'http';

    try {
      // Create Streamable HTTP transport in stateless mode
      // Stateless mode (sessionIdGenerator: undefined) is required for compatibility with mcp-remote
      this.transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
      });

      // Connect MCP Server to transport
      await this.mcpServer.connect(this.transport);
      console.error('[MCP] Server connected to Streamable HTTP transport');

      // Create HTTP or HTTPS server
      if (this.config.https) {
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
          key: fs.readFileSync(keyPath),
        };

        this.server = https.createServer(httpsOptions, this.app);
      } else {
        this.server = http.createServer(this.app);
      }

      // Start listening
      await new Promise<void>((resolve, reject) => {
        this.server!.listen(this.config.port, host, () => {
          console.error(`[${protocol.toUpperCase()}] MCP Server listening on ${protocol}://${host}:${this.config.port}`);
          console.error(`[${protocol.toUpperCase()}] MCP endpoint: ${protocol}://${host}:${this.config.port}/mcp`);
          console.error(`[${protocol.toUpperCase()}] Protocol: Streamable HTTP (2025-03-26)`);
          console.error(`[${protocol.toUpperCase()}] Ready for connections`);
          resolve();
        });

        this.server!.on('error', (error: Error) => {
          console.error(`[${protocol.toUpperCase()}] Server error:`, error);
          reject(error);
        });
      });
    } catch (error) {
      // Cleanup on error
      if (this.transport) {
        await this.transport.close();
        this.transport = null;
      }
      throw error;
    }
  }

  /**
   * Stop the server and close transport
   */
  async stop(): Promise<void> {
    // Close transport first
    if (this.transport) {
      try {
        await this.transport.close();
        console.error('[MCP] Transport closed');
      } catch (error) {
        console.error('[MCP] Error closing transport:', error);
      }
      this.transport = null;
    }

    // Then close HTTP server
    if (this.server) {
      await new Promise<void>((resolve) => {
        this.server!.close(() => {
          console.error('[HTTP] Server stopped');
          resolve();
        });
      });
      this.server = null;
    }
  }

  /**
   * Get the current session ID (if in stateful mode)
   */
  getSessionId(): string | undefined {
    return this.transport?.sessionId;
  }

  /**
   * Check if server is running
   */
  isRunning(): boolean {
    return this.server !== null && this.transport !== null;
  }
}
