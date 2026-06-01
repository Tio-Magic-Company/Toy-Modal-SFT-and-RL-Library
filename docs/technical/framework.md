# Framework Notes

The public SDK routes all runtime work through Modal-backed transports:

```python
service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
)
```

`modal-direct` uses Modal function/class lookup. `http` uses a deployed gateway
URL. The old in-process gateway and public local runtime were removed.

Fast unit tests use test-only fakes for Modal SDK objects and transport handles;
they do not validate deployed behavior.
