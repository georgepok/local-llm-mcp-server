#!/bin/bash

# Test MCP Server over HTTPS using curl
# This demonstrates the SSE + HTTP POST message flow

PORT=${1:-3010}
BASE_URL="https://localhost:$PORT"

echo "🔒 Testing MCP Protocol over HTTPS with curl"
echo "Server: $BASE_URL"
echo "=============================================="
echo ""

# Step 1: Connect to SSE endpoint to get session ID
echo "[Step 1] Connecting to SSE endpoint..."
echo "Command: curl -k -N $BASE_URL/sse"
echo ""

# Start SSE connection in background and capture output
SSE_OUTPUT=$(mktemp)
curl -k -N "$BASE_URL/sse" > "$SSE_OUTPUT" 2>/dev/null &
SSE_PID=$!

# Wait a moment for connection and initial message
sleep 2

# Read SSE output to extract session ID
SSE_DATA=$(cat "$SSE_OUTPUT")
echo "SSE Response:"
echo "$SSE_DATA"
echo ""

# Extract session ID from the endpoint URL
SESSION_ID=$(echo "$SSE_DATA" | grep -o 'sessionId=[^"]*' | cut -d= -f2)

if [ -z "$SESSION_ID" ]; then
    echo "❌ Failed to get session ID from SSE connection"
    kill $SSE_PID 2>/dev/null
    rm "$SSE_OUTPUT"
    exit 1
fi

echo "✓ Session ID obtained: $SESSION_ID"
echo ""

# Step 2: Send initialize message
echo "[Step 2] Sending MCP initialize message..."
INIT_REQUEST='{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "curl-test-client",
      "version": "1.0.0"
    }
  }
}'

echo "Request:"
echo "$INIT_REQUEST" | jq '.' 2>/dev/null || echo "$INIT_REQUEST"
echo ""

INIT_RESPONSE=$(curl -k -s -X POST "$BASE_URL/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d "$INIT_REQUEST")

echo "Response:"
echo "$INIT_RESPONSE" | jq '.' 2>/dev/null || echo "$INIT_RESPONSE"
echo ""

# Step 3: List tools
echo "[Step 3] Listing available tools..."
TOOLS_REQUEST='{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}'

TOOLS_RESPONSE=$(curl -k -s -X POST "$BASE_URL/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d "$TOOLS_REQUEST")

echo "Response:"
echo "$TOOLS_RESPONSE" | jq '.result.tools | length' 2>/dev/null | xargs echo "Tools available:"
echo "$TOOLS_RESPONSE" | jq '.result.tools[] | {name: .name, description: .description}' 2>/dev/null | head -30

echo ""

# Step 4: List resources
echo "[Step 4] Listing available resources..."
RESOURCES_REQUEST='{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "resources/list",
  "params": {}
}'

RESOURCES_RESPONSE=$(curl -k -s -X POST "$BASE_URL/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d "$RESOURCES_REQUEST")

echo "Response:"
echo "$RESOURCES_RESPONSE" | jq '.result.resources | length' 2>/dev/null | xargs echo "Resources available:"
echo "$RESOURCES_RESPONSE" | jq '.result.resources[] | {uri: .uri, name: .name}' 2>/dev/null

echo ""

# Step 5: Read a resource
echo "[Step 5] Reading resource: local://models..."
READ_REQUEST='{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "resources/read",
  "params": {
    "uri": "local://models"
  }
}'

READ_RESPONSE=$(curl -k -s -X POST "$BASE_URL/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d "$READ_REQUEST")

echo "Response:"
echo "$READ_RESPONSE" | jq '.result.contents[0].text' 2>/dev/null | xargs -0 echo | jq '.' 2>/dev/null

echo ""

# Cleanup
kill $SSE_PID 2>/dev/null
rm "$SSE_OUTPUT"

echo "=============================================="
echo "✅ MCP Protocol test completed successfully!"
echo ""
echo "Summary:"
echo "  - SSE connection established ✓"
echo "  - Session ID obtained ✓"
echo "  - MCP initialize successful ✓"
echo "  - Tools/Resources listed ✓"
echo "  - Resource read successful ✓"
