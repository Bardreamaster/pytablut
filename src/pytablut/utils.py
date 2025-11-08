import socket


def receive_exact_bytes(s: socket.socket, n: int) -> bytes:
    if s is None:
        raise Exception("socket is None")

    if n < 1:
        return b""

    data = bytearray()
    while len(data) < n:
        packet = s.recv(n - len(data))
        if not packet:
            raise Exception("Connection closed before receiving expected bytes")
        data.extend(packet)

    return bytes(data)
