import argparse
import subprocess
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--poll-freq", default=1000, type=int, help="Frequency (in ms) to poll clusters"
)
parser.add_argument("hosts", help="File containing hostnames")
args = parser.parse_args()

with open(args.hosts) as fp:
    hosts = []
    for line in fp:
        hosts.append(line.strip())

while True:
    procs = [
        subprocess.Popen(
            [
                "ssh",
                host,
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,power.limit,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "&&",
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for host in hosts
    ]
    for proc in procs:
        proc.wait()

    outputs = [proc.stdout.read().decode() for proc in procs]

    gpu_stats = {}
    node_stats = {
        host: dict(util=0, power_usage=0, memory_usage=0, num_gpus=0, num_procs=0)
        for host in hosts
    }
    cluster_stats = dict(util=0, power_usage=0, memory_usage=0, num_gpus=0, num_procs=0)
    for host, output in zip(hosts, outputs):
        gpu_stats[host] = {}
        for gpu, stats in enumerate(output.splitlines()):
            if "," not in stats:
                node_stats[host]["num_procs"] += 1
                cluster_stats["num_procs"] += 1
                continue

            util, power_draw, power_limit, memory_used, memory_total = map(
                float, stats.split(", ")
            )
            power_usage = 100 * power_draw / power_limit
            memory_usage = 100 * memory_used / memory_total

            gpu_stats[host][gpu] = dict(
                util=util, power_usage=power_usage, memory_usage=memory_usage
            )
            node_stats[host]["util"] += util
            node_stats[host]["memory_usage"] += memory_usage
            node_stats[host]["power_usage"] += power_usage
            node_stats[host]["num_gpus"] += 1
            cluster_stats["util"] += util
            cluster_stats["memory_usage"] += memory_usage
            cluster_stats["power_usage"] += power_usage
            cluster_stats["num_gpus"] += 1

    cluster_stats["util"] /= cluster_stats["num_gpus"]
    cluster_stats["memory_usage"] /= cluster_stats["num_gpus"]
    cluster_stats["power_usage"] /= cluster_stats["num_gpus"]
    for host in hosts:
        node_stats[host]["util"] /= node_stats[host]["num_gpus"]
        node_stats[host]["memory_usage"] /= node_stats[host]["num_gpus"]
        node_stats[host]["power_usage"] /= node_stats[host]["num_gpus"]

    print(f"==={datetime.datetime.now()}")
    print(f"{'name':>10}\t{'util':>10}\t{'power':>10}\t{'memory':>10}\t{'nprocs':>10}")
    print(
        f"{'cluster':>10}\t{cluster_stats['util']:>9.1f}%\t{cluster_stats['power_usage']:>9.1f}%\t{cluster_stats['memory_usage']:>9.1f}%\t{cluster_stats['num_procs']:>10}"
    )
    for host, stats in node_stats.items():
        print(
            f"{host:>10}\t{stats['util']:>9.1f}%\t{stats['power_usage']:>9.1f}%\t{stats['memory_usage']:>9.1f}%\t{stats['num_procs']:>10}"
        )
    print("===")

    time.sleep(args.poll_freq / 1000.0)
