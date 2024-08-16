# Diagnosing Errors

## Hanging

1. Check that the system times on all machines are as close as possible. THis can cause nccl timeouts
2. Check that timeout in `dist.init_process_group(timeout=...)` is sufficiently large.
3. Use `py-spy` to get a dump of stacktraces from all python threads in a running python program.

## System metrics to watch for

1. GPU Power Usage

## Model metrics to watch for

1. Weight norm (diverging)