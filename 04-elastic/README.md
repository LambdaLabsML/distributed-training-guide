# Elastic workers

Elastic meaning workers that can restart

## Considerations

- `rank`, `local_rank`, and `world_size` are all not stable
- How to determine if a node has a problem that simply restarting *can* fix.