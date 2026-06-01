# Transports

`toy_modal` clients use a transport abstraction behind the public SDK surface.
Production runtime paths are Modal-backed.

## modal-direct

`modal-direct` calls a deployed Modal app using the Modal Python client:

```python
service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
)
```

This is the default transport. It requires Modal authentication and a deployed
backend. Calls use Modal function/class lookup and return `APIFuture` handles
for long-running work.

## http

HTTP mode talks to a deployed gateway endpoint:

```python
service = tinker.ServiceClient(
    project_id="demo-http",
    transport="http",
    base_url="https://<your-deployed-gateway>",
    api_key="<optional-api-key>",
)
```

Use this when Python Modal SDK lookup is not the desired client boundary. Long
running work must still be submitted as async jobs and retrieved later.

## Removed Local Runtime

The public `local-mock` transport and the in-process HTTP gateway shortcut have
been removed. Fast unit tests use test-only fakes; user workflows should target
Modal.
