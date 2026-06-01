# Clients And Futures

```python
service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
)
training = service.create_lora_training_client(
    tinker.DEFAULT_BASE_MODEL,
    rank=4,
)
sampling = service.create_sampling_client(
    base_model=tinker.DEFAULT_BASE_MODEL,
)
rest = service.create_rest_client()
```

Heavy calls return `APIFuture` handles:

```python
fwdbwd = training.forward_backward(datums, "cross_entropy")
optim = training.optim_step(tinker.AdamParams(learning_rate=1e-4))

loss = fwdbwd.result()
step = optim.result()
```

`APIFuture` can also be awaited in async code. The same shape is used for
`modal-direct` and deployed HTTP transports.
