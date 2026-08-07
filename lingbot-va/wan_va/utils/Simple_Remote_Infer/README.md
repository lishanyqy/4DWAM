# General Server-Client Setup

## /Simple_Remote_Infer/deploy/qwenpi_policy.py

- QwenPiServer: an example model that provides `init` and `infer` methods.

Wrap the loaded model with `WebsocketPolicyServer` and specify the port.

```python
model_server = WebsocketPolicyServer(model, port=8002)
model_server.serve_forever() # Start listening
```

## ./websocket_client_policy.py

The `__main__()` function shows how to create a dummy model that sends environment information to the real model. Replace the original model with `WebsocketClientPolicy` to use it.
