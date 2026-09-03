"""Helper for the remote PTL box (Windows, Xe3 iGPU).

Credentials are read from cm_gguf_gemm_kernel/remote_machine.txt and are never
printed.

Usage:
    python rptl.py run  "<cmd>"                # run in the remote workdir + venv
    python rptl.py raw  "<cmd>"                # run without cd/venv
    python rptl.py py   <local.py> [args...]   # upload + run with the remote venv
    python rptl.py put  <local> [<remote>]     # upload one file into the workdir
    python rptl.py sync <local> [<local> ...]  # upload several into the workdir
"""
import os
import re
import sys

import paramiko

INFO = "/mnt/river/ovmx/cm_gguf_gemm_kernel/remote_machine.txt"


def creds():
    txt = open(INFO, encoding="utf-8", errors="replace").read()
    ip = re.search(r"IP:\s*(\S+)", txt).group(1)
    user = re.search(r"username:\s*(\S+)", txt).group(1)
    pwd = re.search(r"password:\s*(\S+)", txt).group(1)
    wd = re.search(r"Working directory:\s*(\S+)", txt).group(1)
    act = re.search(r"Python setup:\s*(\S+)", txt).group(1)
    if "\\" in user:
        user = user.split("\\")[-1]
    return ip, user, pwd, wd, act


def connect():
    ip, user, pwd, wd, act = creds()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(ip, username=user, password=pwd, timeout=30,
                allow_agent=False, look_for_keys=False)
    return cli, wd, act


def run(cmd, timeout=14400, quiet=False):
    cli, _, _ = connect()
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    cli.close()
    if not quiet:
        sys.stdout.write(o)
        if e.strip():
            sys.stderr.write("\n[stderr]\n" + e)
    return rc, o, e


def put(local, remote=None):
    cli, wd, _ = connect()
    sftp = cli.open_sftp()
    remote = remote or (wd + "\\" + os.path.basename(local))
    sftp.put(local, remote)
    sftp.close()
    cli.close()
    return remote


def wrapped(cmd):
    _, wd, act = connect()
    return f'cmd /c "call {act}.bat && cd /d {wd} && {cmd}"'


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    sub = sys.argv[1]
    if sub == "run":
        return run(wrapped(sys.argv[2]))[0]
    if sub == "raw":
        return run(sys.argv[2])[0]
    if sub == "py":
        put(sys.argv[2])
        name = os.path.basename(sys.argv[2])
        return run(wrapped(f'python {name} {" ".join(sys.argv[3:])}'))[0]
    if sub == "put":
        print(put(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
        return 0
    if sub == "sync":
        for f in sys.argv[2:]:
            print(put(f))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
