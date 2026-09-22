"""
Distributed queue server for Mali-Ba training.

Exposes the job and result queues over TCP using Python's multiprocessing.managers,
so actor processes on remote machines can participate in training transparently.

Usage
-----
On the training server (started automatically by train_mali_ba.py --distributed):
    The server thread starts automatically — no need to run this directly.

On a remote actor machine:
    python remote_actors.py --server_host <IP> [--server_port 50000] [--num_actors 8]
"""

from multiprocessing.managers import BaseManager

DEFAULT_PORT = 50000
DEFAULT_AUTHKEY = b'malibatraining2024'


class MaliBaQueueManager(BaseManager):
    pass


def start_server(job_queue, result_queue, shared_config, log_queue=None,
                 host='192.168.0.102', port=DEFAULT_PORT, authkey=DEFAULT_AUTHKEY):
    """
    Start a blocking queue server. Intended to be called in a daemon thread.

    Args:
        job_queue:     The mp.Queue actors pull jobs from.
        result_queue:  The mp.Queue actors push completed trajectories to.
        shared_config: A plain dict of config values remote actors need
                       (game_name, initial_game_params, uct_c, max_simulations,
                       games_per_actor).
        log_queue:     Optional mp.Queue; remote actors relay their stdout/stderr
                       lines here so they appear in the desktop's tee'd log file.
        host:          Interface to bind. '0.0.0.0' accepts all connections.
        port:          TCP port to listen on.
        authkey:       Shared secret bytes for authentication.
    """
    MaliBaQueueManager.register('get_job_queue',    callable=lambda: job_queue)
    MaliBaQueueManager.register('get_result_queue', callable=lambda: result_queue)
    MaliBaQueueManager.register('get_config',       callable=lambda: shared_config)
    MaliBaQueueManager.register('get_log_queue',    callable=lambda: log_queue)

    manager = MaliBaQueueManager(address=(host, port), authkey=authkey)
    server = manager.get_server()
    print(f"[QueueServer] Listening on {host}:{port}", flush=True)
    server.serve_forever()  # Blocks forever — run in a daemon thread


def connect_client(host, port=DEFAULT_PORT, authkey=DEFAULT_AUTHKEY):
    """
    Connect to a running queue server.

    Returns the connected manager proxy. Call get_job_queue(), get_result_queue(),
    and get_config() on the returned object to get proxies to each resource.
    """
    MaliBaQueueManager.register('get_job_queue')
    MaliBaQueueManager.register('get_result_queue')
    MaliBaQueueManager.register('get_config')
    MaliBaQueueManager.register('get_log_queue')

    manager = MaliBaQueueManager(address=(host, port), authkey=authkey)
    manager.connect()
    return manager
