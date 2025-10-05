#!/bin/bash

# Simple MCP test using curl
# Shows basic MCP protocol flow over HTTPS

PORT=${1:-3010}
BASE_URL="https://localhost:$PORT"

echo "🔒 MCP Server Test (HTTPS)"
echo "Server: $BASE_URL"
echo "======================================"
echo ""

# Test 1: Health Check
echo "1️⃣  Health Check:"
curl -k -s "$BASE_URL/health" | jq '.'
echo ""

# Test 2: Server Info
echo "2️⃣  Server Info:"
curl -k -s "$BASE_URL/" | jq '.'
echo ""

# Test 3: SSE Connection (shows session ID)
echo "3️⃣  SSE Connection (3 second sample):"
timeout 3 curl -k -N -s "$BASE_URL/sse" 2>/dev/null || true
echo ""
echo ""

# Test 4: Extract session and send message
echo "4️⃣  Full MCP Flow:"
echo ""

# Get session ID
SSE_RESPONSE=$(timeout 2 curl -k -N -s "$BASE_URL/sse" 2>/dev/null || true)
SESSION_ID=$(echo "$SSE_RESPONSE" | grep -o 'sessionId=[^"]*' | cut -d= -f2)

if [ -n "$SESSION_ID" ]; then
    echo "✓ Session ID: $SESSION_ID"
    echo ""

    # Send initialize
    echo "Sending initialize..."
    curl -k -s -X POST "$BASE_URL/message?sessionId=$SESSION_ID" \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
          "protocolVersion": "2024-11-05",
          "capabilities": {},
          "clientInfo": {"name": "curl-test", "version": "1.0.0"}
        }
      }'
    echo ""
    echo ""

    # Request tools list
    echo "Requesting tools/list..."
    curl -k -s -X POST "$BASE_URL/message?sessionId=$SESSION_ID" \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
      }'
    echo ""
    echo ""

    echo "✓ Messages sent successfully"
    echo "  (Responses are sent via SSE stream)"
else
    echo "❌ Could not get session ID"
fi

echo ""
echo "======================================"
echo "✅ Test completed"
echo ""
echo "💡 To see server logs:"
echo "   tail -f /tmp/mcp-server.log"
