import argparse
import socket
import pickle
import struct
import random
import os
import time
import torch

from model.data import loader
from model.server import server
from model.client import client
from model.plot import plot


def send_msg(sock, data):
    packed_data = pickle.dumps(data)
    sock.sendall(struct.pack('>I', len(packed_data)) + packed_data)

def recv_msg(sock):
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    data = b''
    while len(data) < msglen:
        packet = sock.recv(msglen - len(data))
        if not packet:
            return None
        data += packet
    return pickle.loads(data)


def run_fl_server(args):
    print('Initialize Dataset for Server Validation...')
    data_loader = loader('mnist', batch_size=args.batch_size)

    print('Initialize Global Server...')
    s = server(size=args.n_client,
               data_loader=data_loader.get_loader([]))

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', args.port))
    server_socket.listen(args.n_client)
    print(f'[Server] 포트 {args.port}에서 클라이언트 {args.n_client}개 연결 대기 중...')

    client_sockets = []
    for _ in range(args.n_client):
        conn, addr = server_socket.accept()
        print(f'[Server] 클라이언트 접속 성공: {addr}')
        client_sockets.append(conn)

    round_times = []
    total_start = time.time()

    for e in range(args.epoch):
        round_start = time.time()
        print('\n================== Round {:>3} =================='.format(e + 1))

        global_state = s.model.state_dict()
        global_state_cpu = {k: v.cpu() for k, v in global_state.items()}

        for conn in client_sockets:
            send_msg(conn, global_state_cpu)

        clients_info = []
        for conn in client_sockets:
            client_info = recv_msg(conn)
            clients_info.append(client_info)

        s.aggregate_network_mode(clients_info)

        round_elapsed = time.time() - round_start
        round_times.append(round_elapsed)
        print(f'[Round {e+1}]  소요 시간: {round_elapsed:.2f}초')
        torch.save(round_times, './cache/round_times.pkl')

    total_elapsed = time.time() - total_start
    torch.save(total_elapsed, './cache/total_time.pkl')
    print(f'\n전체 학습 소요 시간: {total_elapsed:.2f}초 ({total_elapsed/60:.2f}분)')

    print("학습 종료. 결과를 플로팅합니다.")
    plot()


def run_fl_client(args):
    print(f'Initialize Dataset for Client Rank {args.rank}...')
    data_loader = loader('mnist', batch_size=args.batch_size)

    random.seed(19201077 + 19950920 + args.rank)
    c = client(rank=args.rank,
               data_loader=data_loader.get_loader(random.sample(range(0, 10), 4)),
               local_epoch=args.local_epoch)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f'[Client {args.rank}] 서버({args.server_ip}:{args.port})로 연결 시도 중...')
    client_socket.connect((args.server_ip, args.port))
    print(f'[Client {args.rank}] 연결 완료.')

    for e in range(args.epoch):
        global_state = recv_msg(client_socket)
        if global_state is None:
            break

        local_info = c.run_network_mode(global_state)
        send_msg(client_socket, local_info)

    client_socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed FedAvg with Docker")
    parser.add_argument('--role', type=str, required=True, choices=['server', 'client'])
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_client', type=int, default=2)

    args = parser.parse_args()

    if args.role == 'server':
        run_fl_server(args)
    elif args.role == 'client':
        run_fl_client(args)