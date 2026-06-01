# Checkpointing And Resume

Save training state:

```python
checkpoint = training.save_state("step-1").result()
print(checkpoint.path)
```

Load weights only and reset optimizer state:

```python
training = service.create_training_client_from_state(checkpoint.path)
```

Load weights and optimizer state:

```python
training = service.create_training_client_from_state_with_optimizer(checkpoint.path)
```

Export sampler weights:

```python
sampler = training.save_weights_and_get_sampling_client("step-1-sampler")
```

Training checkpoints use `toy-modal://.../checkpoints/...`. Sampler artifacts
use `toy-modal://.../sampler_weights/...`.

In deployed Modal mode, workers commit Volume writes after checkpoint and
optimizer changes. Other workers reload Volumes before reading those artifacts.
