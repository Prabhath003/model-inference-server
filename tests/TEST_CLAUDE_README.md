# Claude API Test Scripts

Test scripts for the Claude API endpoint in ModelServer.

## Prerequisites

1. **ModelServer Running**: Ensure the ModelServer is running on port 1121
2. **AWS Credentials**: Ensure AWS credentials are configured for Bedrock access
3. **Python Dependencies**: `requests` library installed

## Test Scripts

### 1. Simple Test (`test_claude_simple.py`)

Quick single request test.

```bash
python test_claude_simple.py
```

**What it does:**
- Sends a single request to Claude
- Displays the response
- Good for quick validation

### 2. Full Test Suite (`test_claude.py`)

Comprehensive test suite with 5 different test scenarios.

```bash
python test_claude.py
```

**Test Cases:**

1. **Simple Text Request**: Basic hello message
2. **Multi-turn Conversation**: Test conversation context
3. **Longer Generation**: Creative writing (haiku)
4. **Custom Max Retries**: Request with higher retry limit
5. **Parallel Requests**: 5 concurrent requests to test load balancing

## Request Format

```python
payload = {
    "type": "claude",                                          # Worker type
    "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0", # Claude model ID
    "payload": {
        "messages": [                                          # Conversation messages
            {"role": "user", "content": "Your prompt here"},
            {"role": "assistant", "content": "Previous response"},  # Optional
            {"role": "user", "content": "Follow-up question"}       # Optional
        ],
        "max_tokens": 100,                                     # Optional (default: 4000)
        "temperature": 0.1                                     # Optional (default: 0.1)
    },
    "timeout": 60,                                             # Optional (default: 300)
    "max_retries": 3                                           # Optional (default: 3)
}
```

## Response Format

### Success (200):
```json
{
    "response": "Claude's response text here"
}
```

### Error (500):
```json
{
    "error": "Failed to get response"
}
```

## Features Tested

✅ **Basic Communication**: Simple request/response
✅ **Context Handling**: Multi-turn conversations
✅ **Retry Logic**: Custom retry configurations
✅ **Load Balancing**: Parallel requests
✅ **Throttling**: Pre-emptive backoff behavior
✅ **Error Handling**: AWS Bedrock error management

## Throttling Behavior

The Claude worker implements shared throttling state:

- **Pre-emptive Backoff**: Workers wait before making requests if system is throttled
- **Exponential Backoff**: Delay increases with consecutive throttles (capped at 32s)
- **Shared State**: All workers coordinate backoff together
- **Auto-Reset**: Resets on success or after 60s of no throttling

### Backoff Schedule:
- 1st throttle: ~1s delay
- 2nd throttle: ~2s delay
- 3rd throttle: ~4s delay
- 4th throttle: ~8s delay
- 5th throttle: ~16s delay
- 6th+ throttle: ~32s delay (capped)

## Troubleshooting

### Connection Error
```
Cannot connect to ModelServer at http://localhost:1121
```
**Solution**: Start the ModelServer first

### AWS Credentials Error
```
Could not connect to the endpoint URL
```
**Solution**: Configure AWS credentials with Bedrock access

### Throttling
```
ThrottlingException: Rate exceeded
```
**Solution**: The worker will automatically back off. Reduce request rate or increase workers.

### Timeout
```
Failed to get response
```
**Solution**: Increase timeout value or check AWS region latency

## Example Usage

### Quick Test:
```bash
./test_claude_simple.py
```

### Full Suite:
```bash
./test_claude.py
```

### Custom Test:
```python
import requests

response = requests.post("http://localhost:1121/infer", json={
    "type": "claude",
    "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "payload": {
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
})

print(response.json()['response'])
```

## Performance Notes

- **Worker Scaling**: Automatically scales up to 20% of CPU cores
- **Queue Management**: Dynamic worker allocation based on queue size
- **Shared Throttling**: All workers coordinate to respect rate limits
- **Retry Strategy**: Configurable per-request with max_retries
- **Timeout Handling**: Prevents hanging requests

## Logs

Check logs for detailed execution:
```bash
tail -f logs/model_server_old.log
```

Look for:
- Worker creation: `Created claude worker for ...`
- Throttling: `Pre-emptive backoff due to shared throttling...`
- Success: `Request succeeded, resetting shared throttle counter...`
- Errors: `Claude API error (...): ...`
